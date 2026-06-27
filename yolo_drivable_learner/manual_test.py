import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/segment/drivable_v1/weights/best.pt"
TEST_DIR   = Path("dataset_processed/manual_test")
CONF       = 0.25
WIN_NAME   = "drivable segmentation test"

model  = YOLO(MODEL_PATH)
images = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

print(f"테스트 이미지: {len(images)}장")
print("조작: 다음 [→/Space] | 이전 [←] | 종료 [Q/ESC]")

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

idx = 0
while 0 <= idx < len(images):
    img_path = images[idx]
    results  = model(str(img_path), conf=CONF, verbose=False)
    annotated = results[0].plot()

    cv2.setWindowTitle(WIN_NAME, f"[{idx+1}/{len(images)}] {img_path.name}")
    cv2.imshow(WIN_NAME, annotated)

    key = cv2.waitKey(0) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key in (83, 32, ord('d')):
        idx += 1
    elif key in (81, ord('a')):
        idx = max(0, idx - 1)

cv2.destroyAllWindows()
print("종료")