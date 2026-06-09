# yolo_thread.py
import threading
import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DUMMY = [('NO_OBJECT', 0.0, 0.0)]


class YoloSignalThread(threading.Thread):
    def __init__(self, camera, engine_path: str, conf_thr: float, n_frame: int):
        super().__init__(daemon=True, name="YoloSignalThread")
        self.camera    = camera
        self.conf_thr  = conf_thr
        self.n_frame   = n_frame
        self.running   = False
        self._result   = DUMMY
        self._lock     = threading.Lock()
        self._model    = YOLO(engine_path, task='detect')
        logger.info(f"[YOLO] loaded: {engine_path}")

    def _verify_direction(self, frame, box, direction):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False

        # 1. 그레이스케일 및 이진화
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # 임계값을 높여 노이즈 제거

        # 2. 가로 방향 무게 중심 (x 좌표의 무게 중심)
        # x_coords: 각 열의 인덱스 배열 [0, 1, 2, ..., w]
        # column_sums: 각 열별 흰색 픽셀 합
        column_sums = np.sum(thresh, axis=0) 
        x_coords = np.arange(thresh.shape[1])
        
        # 총 픽셀 합이 0이면 판단 불가
        if np.sum(column_sums) == 0: return False
        
        # 가로축 무게 중심 = (sum(x * pixel_val) / sum(pixel_val))
        cx = np.sum(x_coords * column_sums) / np.sum(column_sums)
        cx_ratio = cx / thresh.shape[1]
        
        logger.debug(f"[CV Geometry Debug] {direction} | CX Ratio: {cx_ratio:.4f}")
        
        # 3. 경계값 판별 (0.5를 기준으로 확실하게 나뉨)
        if direction == 'sign_right':
            return cx_ratio > 0.52 # 우회전은 무게중심이 오른쪽으로 확실히 쏠림
        else: # sign_left
            return cx_ratio < 0.48 # 좌회전은 무게중심이 왼쪽으로 확실히 쏠림

    def run(self):
        self.running = True
        frame_count  = 0
        logger.info("[YOLO] started")

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            frame_count += 1
            if frame_count % self.n_frame != 0:
                continue

            results = self._model(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_name = results.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                if conf < self.conf_thr:
                    continue
                
                # [적용] 좌우 표지판 검증 로직
                #if cls_name in ['sign_left', 'sign_right']:
                    #if not self._verify_direction(frame, box, cls_name):
                        #continue # 가짜(방향 오류)면 통과시킴

                # 결과 저장 (이름, 면적, 중심X)
                area = float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                center_x = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                detections.append((cls_name, area, center_x))
                logger.debug(f"[YOLO Debug] Detected: {cls_name}, Conf: {conf:.2f}, Size: {area:.0f}") 

            detections = sorted(detections, key=lambda d: d[1], reverse=True)

            with self._lock:
                self._result = detections if detections else DUMMY

    def get_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self.running = False
        self.join()
        logger.info("[YOLO] stopped")