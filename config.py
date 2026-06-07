
# ── Camera ────────────────────────────────────────────────────
CAMERA_ID    = 0
FPS          = 30
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── YOLO SIGNAL──────────────────────────────────────────────────────
YOLO_SIGNAL_PATH  = "models/signal_yolo8n.engine"
YOLO_SIGNAL_N_FRAME      = 3       # N프레임마다 1회 추론
YOLO_SIGNAL_CONF_THR  = 0.5

# ── YOLO DRIVABLE──────────────────────────────────────────────────────
DRIVABLE_ENGINE_PATH = "models/drivable_yolo8n_seg.engine"
YOLO_DRIVABLE_N_FRAME      = 1       # N프레임마다 1회 추론
YOLO_DRIVABLE_CONF_THR  = 0.89

# ── Lane Detector ─────────────────────────────────────────────
LANE_ENGINE_PATH  = "models/mobilenet.engine"

# ── Decision ──────────────────────────────────────────────────
DECISION_QUEUE_TIMEOUT = 0.1   # seconds


# ======================== lane_thread ========================
LANE_ENGINE_PATH = "models/center_model.engine"
CONF_THR         = 0.3
Y_RATIO          = 0.85
RATIO_OFFSET     = -0.17


# ======================== yolo_thread ========================
YOLO_CONF_THR = 0.5

# ======================== decision thread ========================
DECISION_INTERVAL = 1/FPS  # 20Hz


