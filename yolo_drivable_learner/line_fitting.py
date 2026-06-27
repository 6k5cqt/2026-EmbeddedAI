import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/segment/drivable_v1/weights/best.pt"
TEST_DIR   = Path("dataset_processed/manual_test")
CONF       = 0.25
WIN_NAME   = "drivable segmentation test"

# ── 파라미터 ───────────────────────────────────
ROI_TOP_RATIO    = 0.2
ROI_BOTTOM_RATIO = 0.9
TARGET_Y_RATIO   = 0.3  
MIN_SEG_WIDTH    = 20
MIN_MASK_RATIO   = 0.25
IMG_CENTER_TOL   = 0.4   # 이미지 너비 기준 중앙 허용 범위
# ───────────────────────────────────────────────

model  = YOLO(MODEL_PATH)
images = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

print(f"테스트 이미지: {len(images)}장")
print("조작: 다음 [→/Space] | 이전 [←] | 종료 [Q/ESC]")

def get_mask(result, shape):
    if result.masks is None:
        return None
    mask = result.masks.data[0].cpu().numpy()
    return cv2.resize(mask, (shape[1], shape[0])) > 0.5

def largest_segment_bounds(row_mask):
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

def draw_centerline(frame, mask):
    h, w = mask.shape

    mask_ratio = mask.sum() / (h * w)
    cv2.putText(frame, f"mask: {mask_ratio:.3f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    if mask_ratio < MIN_MASK_RATIO:
        cv2.putText(frame, "STOP: drivable area too small", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return

    # 마스크 bounding box 기준 ROI
    rows = np.any(mask, axis=1)
    if not rows.any():
        return
    y_min, y_max = np.where(rows)[0][[0, -1]]
    box_h    = y_max - y_min
    y_start  = int(y_min + box_h * ROI_TOP_RATIO)
    y_end    = int(y_min + box_h * ROI_BOTTOM_RATIO)
    target_y = int(y_min + box_h * TARGET_Y_RATIO)

    pts = []
    center_x = w // 2
    for y in range(y_start, y_end, 10):
        result = largest_segment_bounds(mask[y])
        if result is None:
            continue
        lx, rx = result
        if rx - lx < MIN_SEG_WIDTH:
            continue
        cx = (lx + rx) // 2
        if abs(cx - center_x) > w * IMG_CENTER_TOL:
            continue
        pts.append((cx, y))
        cv2.circle(frame, (cx, y), 3, (0, 255, 0), -1)

    cv2.line(frame, (0, y_start), (w, y_start), (100, 100, 100), 1)
    cv2.line(frame, (0, y_end),   (w, y_end),   (100, 100, 100), 1)

    # 중앙 허용 범위 표시
    cv2.line(frame, (int(center_x - w * IMG_CENTER_TOL), 0),
                    (int(center_x - w * IMG_CENTER_TOL), h), (50, 50, 50), 1)
    cv2.line(frame, (int(center_x + w * IMG_CENTER_TOL), 0),
                    (int(center_x + w * IMG_CENTER_TOL), h), (50, 50, 50), 1)

    if len(pts) < 5:
        cv2.putText(frame, "insufficient points", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return

    xs_arr = np.array([p[0] for p in pts], dtype=float)
    ys_arr = np.array([p[1] for p in pts], dtype=float)
    poly   = np.poly1d(np.polyfit(ys_arr, xs_arr, 2))  # 1차 함수

    curve_pts = []
    for y in range(y_start, y_end, 5):
        x = int(poly(y))
        if 0 <= x < w:
            curve_pts.append((x, y))
    for i in range(len(curve_pts) - 1):
        cv2.line(frame, curve_pts[i], curve_pts[i+1], (0, 0, 255), 2)

    target_x = int(np.clip(poly(target_y), 0, w - 1))
    offset   = (target_x - center_x) / (w // 2)

    cv2.circle(frame, (target_x, target_y), 10, (255, 0, 0), -1)
    cv2.line(frame, (center_x, target_y), (target_x, target_y), (255, 255, 0), 2)
    cv2.putText(frame, f"offset: {offset:+.3f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

idx = 0
while 0 <= idx < len(images):
    img_path = images[idx]
    results  = model(str(img_path), conf=CONF, verbose=False)
    frame    = results[0].plot()
    mask     = get_mask(results[0], frame.shape[:2])

    if mask is not None:
        draw_centerline(frame, mask)
    else:
        cv2.putText(frame, "STOP: no detection", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.setWindowTitle(WIN_NAME, f"[{idx+1}/{len(images)}] {img_path.name}")
    cv2.imshow(WIN_NAME, frame)

    key = cv2.waitKey(0) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key in (83, 32, ord('d')):
        idx += 1
    elif key in (81, ord('a')):
        idx = max(0, idx - 1)

cv2.destroyAllWindows()
print("종료")