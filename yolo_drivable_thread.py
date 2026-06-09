import threading
import logging
import cv2
import numpy as np
import time
from ultralytics import YOLO
from flask import Flask, Response

logger = logging.getLogger(__name__)

# ── 웹 스트리밍용 Flask 및 전역 버퍼 설정 ──
app = Flask(__name__)
debug_frame = None
debug_lock = threading.Lock()

# ── 파라미터 ───────────────────────────────────
ROI_TOP_RATIO  = 0.2
ROI_BOT_RATIO  = 0.9
TARGET_Y_RATIO = 0.3

MIN_SEG_WIDTH  = 20
MIN_MASK_RATIO = 0.15
CENTER_TOL     = 0.4
OFFSET_SCALE = 2.0  # offset 스케일 조정
# ───────────────────────────────────────────────

DUMMY = {'state': 'no_detection', 'offset': 0.0, 'mask_ratio': 0.0}


class YoloDrivableThread(threading.Thread):
    def __init__(self, camera, engine_path: str, conf: float = 0.25, n_frame: int = 1):
        super().__init__(daemon=True, name="DrivableThread")
        self.camera  = camera
        self.conf    = conf
        self.n_frame = n_frame
        self.running = False
        self._result = DUMMY
        self._lock   = threading.Lock()
        self._model  = YOLO(engine_path, task='segment')
        logger.info(f"[DRIVABLE] loaded: {engine_path}")

    def _get_mask(self, result, shape):
        if result.masks is None:
            return None
        mask = result.masks.data[0].cpu().numpy()
        return cv2.resize(mask, (shape[1], shape[0])) > 0.5

    def _largest_segment_bounds(self, row_mask):
        segs, in_seg, start = [], False, 0
        for i, v in enumerate(row_mask):
            if v and not in_seg:
                start, in_seg = i, True
            elif not v and in_seg:
                segs.append((start, i - 1))
                in_seg = False
        if in_seg:
            segs.append((start, len(row_mask) - 1))
        if not segs:
            return None
        return max(segs, key=lambda s: s[1] - s[0])

    def _compute(self, mask):
        h, w       = mask.shape
        mask_ratio = mask.sum() / (h * w)

        if mask_ratio < MIN_MASK_RATIO:
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None

        rows = np.any(mask, axis=1)
        if not rows.any():
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        box_h    = y_max - y_min
        y_start  = int(y_min + box_h * ROI_TOP_RATIO)
        y_end    = int(y_min + box_h * ROI_BOT_RATIO)
        target_y = int(y_min + box_h * TARGET_Y_RATIO)
        center_x = w // 2

        pts = []
        for y in range(y_start, y_end, 10):
            r = self._largest_segment_bounds(mask[y])
            if r is None:
                continue
            lx, rx = r
            if rx - lx < MIN_SEG_WIDTH:
                continue
            cx = (lx + rx) // 2
            if abs(cx - center_x) > w * CENTER_TOL:
                continue
            pts.append((cx, y))

        if len(pts) < 5:
            return {'state': 'insufficient', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None

        xs_arr = np.array([p[0] for p in pts], dtype=float)
        ys_arr = np.array([p[1] for p in pts], dtype=float)
        poly   = np.poly1d(np.polyfit(ys_arr, xs_arr, 2))

        target_x = int(np.clip(poly(target_y), 0, w - 1))
        offset = (target_x - center_x) / (w // 2) * OFFSET_SCALE

        # 제어 스레드(DecisionThread) 파손을 막기 위해 기존 딕셔너리와 시각화용 변수들을 함께 리턴
        return {'state': 'ok', 'offset': float(offset), 'mask_ratio': float(mask_ratio)}, poly, target_y, target_x

    def run(self):
        self.running = True
        frame_count  = 0
        logger.info("[DRIVABLE] started")

        # ── 이 스레드가 시작될 때 Flask 백그라운드 웹 서버도 같이 시작 ──
        threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
        logger.info("[DRIVABLE] Visualization Web Server started on port 5000")

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            frame_count += 1
            if frame_count % self.n_frame != 0:
                continue

            results = self._model(frame, conf=self.conf, verbose=False)[0]
            mask    = self._get_mask(results, frame.shape[:2])
            
            # 주행 데이터 처리 및 시각화용 데이터 추출
            if mask is not None:
                result, poly, target_y, target_x = self._compute(mask)
            else:
                result, poly, target_y, target_x = DUMMY, None, None, None

            # ── [디버깅용 실시간 시각화 이미지 그리기] ──
            debug_img = frame.copy()
            
            # 1. 주행 가능 영역 초록색 반투명 마스크 표시
            if mask is not None:
                debug_img[mask] = debug_img[mask] * 0.6 + np.array([0, 255, 0], dtype=np.uint8) * 0.4
            
            # 2. 피팅된 2차 함수 라인 그리기
            if poly is not None:
                ys = np.arange(int(frame.shape[0] * ROI_TOP_RATIO), int(frame.shape[0] * ROI_BOT_RATIO), 10)
                xs = poly(ys).astype(int)
                for i in range(len(xs) - 1):
                    if 0 <= xs[i] < frame.shape[1] and 0 <= xs[i+1] < frame.shape[1]:
                        cv2.line(debug_img, (xs[i], ys[i]), (xs[i+1], ys[i+1]), (255, 0, 0), 3) # 파란색 선
                
                # 3. 타겟 지점 빨간색 원 표시
                cv2.circle(debug_img, (target_x, target_y), 10, (0, 0, 255), -1)
                
            # 4. 현재 상태 및 Offset 텍스트 출력
            cv2.putText(debug_img, f"State: {result['state']}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(debug_img, f"Offset: {result['offset']:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 웹서버로 보낼 공유 버퍼 갱신
            with debug_lock:
                global debug_frame
                _, buffer = cv2.imencode('.jpg', debug_img)
                debug_frame = buffer.tobytes()

            logger.debug(f"[DRIVABLE] {result}")
            with self._lock:
                self._result = result

    def get_result(self):
        with self._lock:
            return self._result

    def stop(self):
        self.running = False
        self.join()
        logger.info("[DRIVABLE] stopped")


# ── 웹 브라우저 전송용 Flask MJPEG 라우터 (클래스 외부 배치) ──
def generate():
    while True:
        with debug_lock:
            if debug_frame:
                yield (b'--frame\n'
                       b'Content-Type: image/jpeg\n\n' + debug_frame + b'\n')
        time.sleep(0.03) # 약 30 FPS 수준으로 전송 제한하여 부하 감소

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')