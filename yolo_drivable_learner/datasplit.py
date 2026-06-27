import shutil
import random
import yaml
from pathlib import Path

# ── 설정 ──────────────────────────────────────
SRC_IMG_DIR = Path("dataset/images/Train")
SRC_LBL_DIR = Path("dataset/labels/Train")
OUT_DIR     = Path("dataset_processed")
VAL_RATIO   = 0.2
SEED        = 42
CLASS_NAMES = ["drivable"]
# ───────────────────────────────────────────────

random.seed(SEED)

image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
labeled, unlabeled = [], []

for img_path in SRC_IMG_DIR.iterdir():
    if img_path.suffix.lower() not in image_exts:
        continue
    lbl_path = SRC_LBL_DIR / f"{img_path.stem}.txt"
    if lbl_path.exists():
        labeled.append(img_path.stem)
    else:
        unlabeled.append(img_path.stem)

print(f"전체 이미지 : {len(labeled) + len(unlabeled)}")
print(f"라벨 있는   : {len(labeled)}")
print(f"라벨 없는   : {len(unlabeled)}  → manual_test/")

# 2. train / val 분할
random.shuffle(labeled)
val_n  = int(len(labeled) * VAL_RATIO)
splits = {
    "val"  : labeled[:val_n],
    "train": labeled[val_n:]
}
print(f"train: {len(splits['train'])}장  |  val: {len(splits['val'])}장")

# 3. train/val 복사
for split, stems in splits.items():
    img_out = OUT_DIR / "images" / split
    lbl_out = OUT_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        for ext in image_exts:
            src_img = SRC_IMG_DIR / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy(src_img, img_out / src_img.name)
                break
        shutil.copy(SRC_LBL_DIR / f"{stem}.txt", lbl_out / f"{stem}.txt")

# 4. 라벨 없는 이미지 → manual_test/
manual_dir = OUT_DIR / "manual_test"
manual_dir.mkdir(parents=True, exist_ok=True)
for stem in unlabeled:
    for ext in image_exts:
        src_img = SRC_IMG_DIR / f"{stem}{ext}"
        if src_img.exists():
            shutil.copy(src_img, manual_dir / src_img.name)
            break
print(f"manual_test: {len(unlabeled)}장 복사 완료")

# 5. data.yaml 생성
data_yaml = {
    "path"  : str(OUT_DIR.resolve()),
    "train" : "images/train",
    "val"   : "images/val",
    "nc"    : len(CLASS_NAMES),
    "names" : CLASS_NAMES
}
with open(OUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

print(f"\n완료 → {OUT_DIR / 'data.yaml'}")