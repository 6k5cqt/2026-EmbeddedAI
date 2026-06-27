from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n-seg.pt")

    model.train(
    model="yolo26n-seg.pt",
    data="dataset_processed/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    workers=0,

    # 추가
    patience=30,          # val mAP 30 epoch 동안 안 오르면 자동 조기 종료
    optimizer="AdamW",    # 소규모 데이터에서 SGD보다 수렴 안정적
    cos_lr=True,          # epoch 진행하면서 lr 부드럽게 감소
    cache=True,           # 이미지 RAM 캐싱 → 학습 속도 향상

    # 기존 augmentation 유지
    degrees=5.0,
    perspective=0.0005,
    shear=2.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.5,
    copy_paste=0.3,
    mosaic=1.0,

    project=".",
    name="drivable_v1_26n",
)