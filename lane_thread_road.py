# lane_thread.py
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import threading
import logging

from model import decode
from config import RATIO_OFFSET, DECISION_INTERVAL

logger = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame: np.ndarray) -> np.ndarray:
    x = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)
    return np.ascontiguousarray(x[np.newaxis, :])


class TRTInference:
    def __init__(self, engine_path: str):
        self._ctx = cuda.Device(0).make_context()

        logger_trt = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger_trt) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._alloc()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        self._ctx.pop()
        logger.info(f"[TRT] loaded: {engine_path}")

    def _alloc(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name   = self.engine.get_tensor_name(i)
            shape  = self.engine.get_tensor_shape(name)
            dtype  = trt.nptype(self.engine.get_tensor_dtype(name))
            host   = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device = cuda.mem_alloc(host.nbytes)
            bindings.append(int(device))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host, 'device': device})
            else:
                outputs.append({'host': host, 'device': device})
        return inputs, outputs, bindings, stream

    def infer(self, x: np.ndarray) -> np.ndarray:
        self._ctx.push()
        try:
            np.copyto(self.inputs[0]['host'], x.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            return self.outputs[0]['host']
        finally:
            self._ctx.pop()


class LaneThread(threading.Thread):
    def __init__(self, camera, engine_path: str,
                 conf_thr: float, y_ratio: float, ratio_offset: float):
        super().__init__(daemon=True, name="LaneThread")
        self.camera       = camera
        self.engine_path  = engine_path
        self.conf_thr     = conf_thr
        self.y_ratio      = y_ratio
        self.ratio_offset = RATIO_OFFSET
        self.running      = False
        self._paused      = False
        self._result      = [0.0]
        self._lock        = threading.Lock()

    def pause(self):
        self._paused = True
        logger.info("[Lane] paused")

    def resume(self):
        self._paused = False
        logger.info("[Lane] resumed")

    def run(self):
        try:
            trt_engine = TRTInference(self.engine_path)
        except Exception as e:
            logger.error(f"[Lane] TRT init failed: {e}")
            return

        self.running = True
        logger.info("[Lane] started")

        while self.running:
            if self._paused:
                time.sleep(DECISION_INTERVAL)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            y_px   = int(h * self.y_ratio)
            output = trt_engine.infer(preprocess(frame)).reshape(2, 64)
            result = decode(output, w, y_px, self.conf_thr)

            if result['centers']:
                offsets = [float(c - w / 2) for c in result['centers']]
                out = [float(o / (w / 2)) + self.ratio_offset for o in offsets]
            else:
                out = [0.0]

            with self._lock:
                self._result = out

    def get_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self.running = False
        self.join()
        logger.info("[Lane] stopped")