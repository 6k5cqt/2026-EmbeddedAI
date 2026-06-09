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
MIN_MASK_RATIO = 0.20
CENTER_TOL     = 0.4
OFFSET_SCALE = 2.0  # offset 스케일 조정
OPEN_FIELD_THRES = 0.80 #이거 아래면 임시차선

# ── 디버그 영상 저장 옵션 ───────────────────────
SAVE     = False                        # True 로 바꾸면 저장
SAVE_DIR = "./debug_output"             # 저장 폴더 (없으면 자동 생성)
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
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None, None, None, None

        rows = np.any(mask, axis=1)
        if not rows.any():
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None, None, None, None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        box_h    = y_max - y_min
        y_start  = int(y_min + box_h * ROI_TOP_RATIO)
        y_end    = int(y_min + box_h * ROI_BOT_RATIO)
        target_y = int(y_min + box_h * TARGET_Y_RATIO)
        center_x = w // 2

        pts = []
        widths = [] # 누적가로폭
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
            widths.append(rx - lx)

        if len(pts) < 5:
            return {'state': 'insufficient', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}, None, None, None, None, None, None

        xs_arr = np.array([p[0] for p in pts], dtype=float)
        ys_arr = np.array([p[1] for p in pts], dtype=float)
        poly   = np.poly1d(np.polyfit(ys_arr, xs_arr, 2))

        target_x = int(np.clip(poly(target_y), 0, w - 1))
        offset = (target_x - center_x) / (w // 2) * OFFSET_SCALE

        avg_width_ratio = np.mean(widths) / w

        if avg_width_ratio > OPEN_FIELD_THRES:
            current_state = 'ok'       # 완전 허허벌판 상황
        else:
            current_state = 'road_ok'  # 양옆이 막혀 차선 형태가 잡힌 상황

        # 제어 스레드(DecisionThread) 파손을 막기 위해 기존 딕셔너리와 시각화용 변수들을 함께 리턴
        return {'state': current_state, 'offset': float(offset), 'mask_ratio': float(mask_ratio)}, poly, target_y, target_x, y_start, y_end, pts

    def run(self):
        self.running = True
        frame_count  = 0
        logger.info("[DRIVABLE] started")

        # ── 이 스레드가 시작될 때 Flask 백그라운드 웹 서버도 같이 시작 ──
        threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
        logger.info("[DRIVABLE] Visualization Web Server started on port 5000")

        # ── 디버그 영상 저장용 VideoWriter 초기화 ──
        _writer = None
        if SAVE:
            import os, datetime
            os.makedirs(SAVE_DIR, exist_ok=True)
            _save_path = None  # 첫 프레임에서 해상도 확인 후 생성

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
                result, poly, target_y, target_x, y_start, y_end, pts = self._compute(mask)
            else:
                result, poly, target_y, target_x, y_start, y_end, pts = DUMMY, None, None, None, None, None, None

            # ── [디버깅용 실시간 시각화 이미지 그리기] (manual_test.py 동일 스타일) ──
            # 베이스: results[0].plot() → YOLO 세그멘테이션 오버레이 포함
            debug_img = results.plot()
            h_f, w_f  = debug_img.shape[:2]
            center_x  = w_f // 2

            # mask_ratio 텍스트 (항상 표시)
            cv2.putText(debug_img, f"mask: {result['mask_ratio']:.3f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            state = result['state']
            if state in ('no_detection', 'stop'):
                label = "STOP: no detection" if state == 'no_detection' else "STOP: drivable area too small"
                cv2.putText(debug_img, label, (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            elif state == 'insufficient':
                cv2.putText(debug_img, "insufficient points", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # ROI 경계선 + 중앙 허용범위선은 그려줌
                if y_start is not None:
                    cv2.line(debug_img, (0, y_start), (w_f, y_start), (100, 100, 100), 1)
                    cv2.line(debug_img, (0, y_end),   (w_f, y_end),   (100, 100, 100), 1)
                    cv2.line(debug_img, (int(center_x - w_f * CENTER_TOL), 0),
                                        (int(center_x - w_f * CENTER_TOL), h_f), (50, 50, 50), 1)
                    cv2.line(debug_img, (int(center_x + w_f * CENTER_TOL), 0),
                                        (int(center_x + w_f * CENTER_TOL), h_f), (50, 50, 50), 1)
                # pts 초록 점
                if pts:
                    for (cx, cy) in pts:
                        cv2.circle(debug_img, (cx, cy), 3, (0, 255, 0), -1)

            else:  # 'ok'
                # 1. ROI 경계 회색 수평선
                cv2.line(debug_img, (0, y_start), (w_f, y_start), (100, 100, 100), 1)
                cv2.line(debug_img, (0, y_end),   (w_f, y_end),   (100, 100, 100), 1)
                # 2. 중앙 허용범위 어두운 수직선
                cv2.line(debug_img, (int(center_x - w_f * CENTER_TOL), 0),
                                    (int(center_x - w_f * CENTER_TOL), h_f), (50, 50, 50), 1)
                cv2.line(debug_img, (int(center_x + w_f * CENTER_TOL), 0),
                                    (int(center_x + w_f * CENTER_TOL), h_f), (50, 50, 50), 1)
                # 3. 피팅 샘플 포인트 초록 점
                for (cx, cy) in pts:
                    cv2.circle(debug_img, (cx, cy), 3, (0, 255, 0), -1)
                # 4. 피팅 곡선 빨간색 step=5
                curve_pts = []
                for y in range(y_start, y_end, 5):
                    x = int(poly(y))
                    if 0 <= x < w_f:
                        curve_pts.append((x, y))
                for i in range(len(curve_pts) - 1):
                    cv2.line(debug_img, curve_pts[i], curve_pts[i+1], (0, 0, 255), 2)
                # 5. 타겟 지점 파란 원 + 중앙→타겟 시안 선
                cv2.circle(debug_img, (target_x, target_y), 10, (255, 0, 0), -1)
                cv2.line(debug_img, (center_x, target_y), (target_x, target_y), (255, 255, 0), 2)
                # 6. offset 텍스트
                if state == 'ok':
                    cv2.putText(debug_img, f"OK (Open Field) offset: {result['offset']:+.3f}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                else:  # 'road_ok'
                    cv2.putText(debug_img, f"ROAD_OK (Lane) offset: {result['offset']:+.3f}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # ── 디버그 영상 저장 ──
            if SAVE:
                if _writer is None:
                    import datetime, os
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _save_path = os.path.join(SAVE_DIR, f"drivable_{ts}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    _writer = cv2.VideoWriter(_save_path, fourcc, 30, (w_f, h_f))
                    logger.info(f"[DRIVABLE] Saving debug video: {_save_path}")
                _writer.write(debug_img)

            # 웹서버로 보낼 공유 버퍼 갱신
            with debug_lock:
                global debug_frame
                _, buffer = cv2.imencode('.jpg', debug_img)
                debug_frame = buffer.tobytes()

            logger.debug(f"[DRIVABLE] {result}")
            with self._lock:
                self._result = result

        # ── 루프 종료 시 VideoWriter 해제 ──
        if SAVE and _writer is not None:
            _writer.release()
            logger.info(f"[DRIVABLE] Saved debug video: {_save_path}")

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