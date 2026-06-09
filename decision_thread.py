import threading
import time
import logging

from ctrl_utils import BaseController

logger = logging.getLogger(__name__)

# ── Params ────────────────────────────────────────────────────
DEBUG_LOG_INTERVAL = 1
DEBUG_CONTROL = False

# 사용 표지판만 정의
NAME_DIR = {
    'LEFT': 'sign_left', 'RIGHT': 'sign_right',
    'STOP': 'sign_stop'
}

# 각 class별 카운팅 임계치
CLASS_SIZE_TR = {
    'LEFT': 5000, 'RIGHT': 5000, 'STOP': 2000,
}
COUNTER_TR = {
    'LEFT': 5, 'RIGHT': 5, 'STOP': 5, 'DEFAULT': 10,
}

# 시나리오 파라미터
DEFAULT_SPEED = 0.2
BACK_PERIOD   = 2.0    # STOP 후 후진 시간
BACK_SPEED    = -0.1   # 후진 속도
TURN_PERIOD   = 1.5    # 좌우회전 강제 주행 시간
LEFT_OFFSET   = -0.5   # 강제 좌회전 오프셋
RIGHT_OFFSET  = 0.5    # 강제 우회전 오프셋

# utils
UART_DEV  = '/dev/ttyUSB0'
BAUD_RATE = 115200

# PID
Kp, Ki, Kd = 0.8, 0.0, 0.2
I_LIMIT = 0.3
INTENSITY_MIN, INTENSITY_MAX = 0.3, 0.5
MARGIN_ERROR = 0.08
INNER_SPEED = 0.05

class DecisionThread(threading.Thread):
    def __init__(self, lane, yolo, drivable, interval: float):
        super().__init__(daemon=True, name="DecisionThread")
        self.lane     = lane
        self.yolo     = yolo
        self.drivable = drivable
        self.interval = interval
        self.running  = False

        self.speed = 0.0
        self.state = 'OFFROAD'
        self.return_val = [0.0, 0.0]
        self._log_counter = 0

        self.prev_time = 0.0
        self.class_counters = {}
        self.detections = []

        self._integral_sum = 0.0
        self._prev_ratio   = 0.0
        self._prev_time    = time.time()

        self._base = BaseController(UART_DEV, BAUD_RATE)

    def _compute_lr(self, offset_ratio: float, speed: float):
        if speed == 0: return 0.0, 0.0
        if speed < 0: return speed, speed # 후진 시 단순 제어

        if abs(offset_ratio) < MARGIN_ERROR:
            self._prev_ratio = 0.0
            self._integral_sum = 0.0
            return speed, speed

        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now

        derivative = (offset_ratio - self._prev_ratio) / dt
        self._prev_ratio = offset_ratio
        self._integral_sum = max(-I_LIMIT, min(I_LIMIT, self._integral_sum + offset_ratio * dt))

        p = Kp * abs(offset_ratio)
        i = Ki * self._integral_sum
        d = Kd * abs(derivative)
        total = p + i + (d if offset_ratio * derivative >= 0 else -d)
        intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, total))

        return (intensity, INNER_SPEED) if offset_ratio > 0 else (INNER_SPEED, intensity)

    def _update_class_counters(self, detections):
        detected = {d[0]: d[1] for d in detections}
        for key, cls in NAME_DIR.items():
            if cls in detected and detected[cls] > CLASS_SIZE_TR[key]:
                self.class_counters[key] = self.class_counters.get(key, 0) + 1
            else:
                self.class_counters[key] = max(0, self.class_counters.get(key, 0) - 1)

    def _check_counter(self):
        for key in NAME_DIR:
            if self.class_counters.get(key, 0) >= COUNTER_TR[key]:
                return key
        return None

    def _process_yolo(self, detections):
        self.detections = detections
        self._update_class_counters(detections)
        triggered = self._check_counter()

        if self.state == 'DEFAULT':
            if triggered == 'STOP':
                self.lane.pause() # 오프로드 백 진입 시 딱 한 번 수행
                self.prev_time = time.time()
                self.state = 'OFFROAD_BACK'
        
        elif self.state == 'OFFROAD':
            if triggered == 'LEFT':
                self.prev_time = time.time()
                self.state = 'OFFROAD_LEFT'
            elif triggered == 'RIGHT':
                self.prev_time = time.time()
                self.state = 'OFFROAD_RIGHT'

    def _process_lane(self, offset_ratios):
        if self.state == 'DEFAULT':
            self.return_val[:] = offset_ratios[-1], DEFAULT_SPEED

        elif self.state == 'OFFROAD_BACK':
            if time.time() - self.prev_time < BACK_PERIOD:
                self.return_val[:] = 0.0, BACK_SPEED
            else:
                self.state = 'OFFROAD'
                self.class_counters = {}

        elif self.state == 'OFFROAD_LEFT':
            if time.time() - self.prev_time < TURN_PERIOD:
                self.return_val[:] = LEFT_OFFSET, DEFAULT_SPEED
            else:
                self.state = 'OFFROAD'
                self.class_counters = {}

        elif self.state == 'OFFROAD_RIGHT':
            if time.time() - self.prev_time < TURN_PERIOD:
                self.return_val[:] = RIGHT_OFFSET, DEFAULT_SPEED
            else:
                self.state = 'OFFROAD'
                self.class_counters = {}

        elif self.state == 'OFFROAD':
            self.return_val[:] = offset_ratios[-1], self.speed
        else:
            raise Exception(f"[Decision] Unknown state: {self.state}")

    def run(self):
        self.running = True
        logger.info("[Decision] started")
        while self.running:
            t0 = time.perf_counter()
            if self.state in ('OFFROAD', 'OFFROAD_BACK', 'OFFROAD_LEFT', 'OFFROAD_RIGHT'):
                result = self.drivable.get_result()
                offset_ratios = [result['offset']] if result['state'] == 'ok' else [0.0]
                if self.state == 'OFFROAD':
                    self.speed = 0.0 if result['state'] in ('stop', 'no_detection') else DEFAULT_SPEED
            else:
                offset_ratios = self.lane.get_result()
                logger.debug(f"[DEBUG] State: {self.state} | Raw Offsets: {offset_ratios}")
                
            detections = self.yolo.get_result()
            if not DEBUG_CONTROL:
                self._process_yolo(detections)
                self._process_lane(offset_ratios)
                self._send(self.return_val[0], self.return_val[1])

            sleep_time = self.interval - (time.perf_counter() - t0)
            if sleep_time > 0: time.sleep(sleep_time)

    def _send(self, offset_ratio: float, speed: float):
        L, R = self._compute_lr(offset_ratio, speed)
        self._base.send_command({"T": 1, "L": -R, "R": -L})