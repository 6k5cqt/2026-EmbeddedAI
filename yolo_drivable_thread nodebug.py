import threading
import logging
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ── 파라미터 ───────────────────────────────────
ROI_TOP_RATIO  = 0.2
ROI_BOT_RATIO  = 0.9
TARGET_Y_RATIO = 0.3

MIN_SEG_WIDTH  = 20
MIN_MASK_RATIO = 0.25
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
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}

        rows = np.any(mask, axis=1)
        if not rows.any():
            return {'state': 'stop', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}

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
            return {'state': 'insufficient', 'offset': 0.0, 'mask_ratio': float(mask_ratio)}

        xs_arr = np.array([p[0] for p in pts], dtype=float)
        ys_arr = np.array([p[1] for p in pts], dtype=float)
        poly   = np.poly1d(np.polyfit(ys_arr, xs_arr, 2))

        target_x = int(np.clip(poly(target_y), 0, w - 1))
        offset = (target_x - center_x) / (w // 2) * OFFSET_SCALE

        return {'state': 'ok', 'offset': float(offset), 'mask_ratio': float(mask_ratio)}

    def run(self):
        self.running = True
        frame_count  = 0
        logger.info("[DRIVABLE] started")

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            frame_count += 1
            if frame_count % self.n_frame != 0:
                continue

            results = self._model(frame, conf=self.conf, verbose=False)[0]
            mask    = self._get_mask(results, frame.shape[:2])
            result  = self._compute(mask) if mask is not None else DUMMY

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