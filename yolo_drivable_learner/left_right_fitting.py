import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

MODEL_PATH = "runs/segment/drivable_v1/weights/best.pt"
TEST_DIR   = Path("dataset_processed/manual_test")
CONF       = 0.25
WIN_NAME   = "drivable segmentation test"

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
    """한 row에서 가장 큰 연속 segment의 좌/우 경계 반환"""
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
    return max(segs, key=lambda s: s[1] - s[0])  # (lx, rx)

def ransac_poly(ys, xs, degree=2):
    pipe = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(residual_threshold=15, random_state=42)
    )
    pipe.fit(ys.reshape(-1, 1), xs)
    return pipe

def draw_overlay(frame, mask):
    h, w = mask.shape

    # 하단 1/3만 사용 (ROI)
    roi_start = h * 2 // 3
    mid_pts = []

    for y in range(roi_start, h, 10):
        result = largest_segment_bounds(mask[y])
        if result is None:
            continue
        lx, rx = result
        if rx - lx < 20:  # 너무 좁은 건 노이즈
            continue
        mid = (lx + rx) // 2
        mid_pts.append((mid, y))
        cv2.circle(frame, (lx, y), 3, (255, 0, 255), -1)   # 왼쪽 경계
        cv2.circle(frame, (rx, y), 3, (255, 0, 255), -1)   # 오른쪽 경계
        cv2.circle(frame, (mid, y), 3, (0, 255, 0), -1)    # 중앙점

    # ROI 표시
    cv2.line(frame, (0, roi_start), (w, roi_start), (100, 100, 100), 1)

    if len(mid_pts) < 5:
        cv2.putText(frame, "insufficient points", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return

    xs_arr = np.array([p[0] for p in mid_pts], dtype=float)
    ys_arr = np.array([p[1] for p in mid_pts], dtype=float)

    try:
        pipe = ransac_poly(ys_arr, xs_arr)
    except Exception:
        return

    # polyfit 곡선
    curve_pts = []
    for y in range(roi_start, h, 5):
        x = int(pipe.predict([[y]])[0])
        if 0 <= x < w:
            curve_pts.append((x, y))
    for i in range(len(curve_pts) - 1):
        cv2.line(frame, curve_pts[i], curve_pts[i + 1], (0, 0, 255), 2)

    # 목표점: ROI 상단 (더 앞을 봄)
    target_y = roi_start + (h - roi_start) // 3
    target_x = int(np.clip(pipe.predict([[target_y]])[0], 0, w - 1))
    center_x = w // 2
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
        draw_overlay(frame, mask)

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