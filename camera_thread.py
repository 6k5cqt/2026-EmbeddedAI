# camera_thread.py
import cv2
import threading
import logging
import time
import os

logger = logging.getLogger(__name__)

DATASET = False  # True로 바꾸면 데이터셋 저장 모드
DATASET_DIR = "dataset/images"
DATASET_INTERVAL = 0.5  # 저장 간격 (초)


def gstreamer_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, "
        f"framerate=(fraction)30/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class CameraThread(threading.Thread):
    def __init__(self, sensor_id: int, fps: int, width: int, height: int):
        super().__init__(daemon=True, name="CameraThread")
        self.fps = fps
        self._interval = 1.0 / fps
        self._frame = None
        self._lock = threading.Lock()
        self.running = False

        pipeline = gstreamer_pipeline(sensor_id, width, height, fps)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            raise RuntimeError(f"CSI Camera {sensor_id} failed to open")

        logger.info(f"[Camera] sensor_id={sensor_id} | {width}x{height} @ {fps}fps")

        if DATASET:
            os.makedirs(DATASET_DIR, exist_ok=True)
            self._dataset_last_saved = 0.0
            logger.info(f"[Dataset] 저장 모드 ON → {DATASET_DIR} / {DATASET_INTERVAL}초 간격")

    def _save_dataset_frame(self, frame):
        now = time.time()
        if now - self._dataset_last_saved >= DATASET_INTERVAL:
            filename = os.path.join(DATASET_DIR, f"{int(now * 1000)}.jpg")
            cv2.imwrite(filename, frame)
            self._dataset_last_saved = now

    def run(self):
        self.running = True

        while self.running:
            t0 = time.perf_counter()

            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

                if DATASET:
                    self._save_dataset_frame(frame)
            else:
                logger.warning("[Camera] frame read failed — skipping")

            elapsed = time.perf_counter() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        self.join()
        self._cap.release()
        logger.info("[Camera] stopped")


if __name__ == "__main__":
    from config import CAMERA_ID, FPS, FRAME_WIDTH, FRAME_HEIGHT

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    cam = CameraThread(
        sensor_id=CAMERA_ID,
        fps=FPS,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
    )
    cam.start()

    print("3초 후 프레임 저장...")
    time.sleep(3)

    frame = cam.get_frame()
    if frame is not None:
        cv2.imwrite("debug_frame.jpg", frame)
        print(f"저장 완료: debug_frame.jpg | shape={frame.shape}")
    else:
        print("프레임 없음 — 카메라 확인 필요")

    cam.stop()