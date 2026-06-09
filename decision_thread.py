import threading
import time
import logging

from ctrl_utils import BaseController
from config import FRAME_WIDTH, RATIO_OFFSET

logger = logging.getLogger(__name__)

# ── Params ────────────────────────────────────────────────────
DEBUG_LOG_INTERVAL = 1
MORE = True #class counter 보고싶을 때 True

# 사용 표지판만 정의
NAME_DIR = {
    'LEFT': 'sign_left', 'RIGHT': 'sign_right', 'PACKET': 'sign_stop', 'BIG_PACKET': 'sign_stop'
}

# 각 class별 카운팅 임계치
CLASS_SIZE_TR = {
    'LEFT': 2500, 'RIGHT': 2500, 'PACKET': 1000, 'BIG_PACKET':5000
}
COUNTER_TR = {
    'LEFT': 5, 'RIGHT': 5, 'DEFAULT': 10, 'EOR':7, 'PACKET': 5, 'BIG_PACKET':5
}

# 시나리오 파라미터
DEFAULT_SPEED = 0.2
EOR_TURN_SPEED = 0.37

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

        self.result = {}
        self.state = 'OFFROAD'
        self.return_val = [0.0, 0.0, 'normal']
        self._log_counter = 0
        self.triggered = ''
        self.travere_counter = 0

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

    def _check_one_counter(self):
        self.triggered = {key: self.class_counters.get(key, 0)
                    for key in NAME_DIR
                    if self.class_counters.get(key, 0) >= COUNTER_TR[key]}
        return max(self.triggered, key=self.triggered.get) if self.triggered else None
    

    def _check_all_counter(self):
        self.triggered = {key: self.class_counters.get(key, 0)
                    for key in NAME_DIR
                    if self.class_counters.get(key, 0) >= COUNTER_TR[key]}
        return list(self.triggered.keys()) if self.triggered else []

    def _traverse(self, condition: bool, target_state: str, reset=True):
        if condition:
            self.state = target_state
            if reset: self.class_counters = {}

    def _process_yolo(self):
        self.detections = self.yolo.get_result()
        self._update_class_counters(self.detections)
        triggered = self._check_one_counter()
        triggered_all = self._check_all_counter()

        if self.state == 'OFFROAD':
            self._traverse(triggered == 'RIGHT', 'EOR_RIGHT')
            self._traverse(triggered == 'LEFT', 'EOR_LEFT')
            self._traverse('PACKET' in triggered_all, 'GOING_PAKCKET')
        if self.state == 'GOING_PAKCKET':
            self._traverse('BIG_PACKET' in triggered_all, 'LOADING', False)


    def _count_traverse(self, condition):
        if condition:
            self.travere_counter = self.travere_counter + 1
        else:
            self.travere_counter = max(0, self.travere_counter - 1)

    # target이 없으면 True, False반환
    def _is_traverse(self, threshold, target=''):
        if target == '':
            if self.travere_counter > threshold:
                self.travere_counter = 0
                return True
            else:
                return False
        else:
            if self.travere_counter > threshold:
                self.travere_counter = 0
                self.state = target
                return True
            else:
                return False
            
    def _process_lane(self):
        self.result = self.drivable.get_result()
        if self.state == 'OFFROAD':
            offset_ratios = [self.result['offset']] if self.result['state'] in ('ok', 'road_ok') else [0.0]
            speed = 0.0 if self.result['state'] in ('stop', 'no_detection') else DEFAULT_SPEED
            self.return_val[:] = offset_ratios[-1], speed, 'normal'
        elif self.state == 'EOR_LEFT':
            # drivable 찾을 때까지 좌회전
            self.return_val[:] = -EOR_TURN_SPEED, DEFAULT_SPEED, 'spin'
            # 길 찾을 때까지 카운터 쌓고 traverse
            self._count_traverse(self.result['state'] == 'road_ok')
            self._is_traverse(threshold=5, target='OFFROAD')
        elif self.state == 'EOR_RIGHT':
            self.return_val[:] = +EOR_TURN_SPEED, DEFAULT_SPEED, 'spin'
            self._count_traverse(self.result['state'] == 'road_ok')
            self._is_traverse(threshold=5, target='OFFROAD')
        elif self.state == 'GOING_PAKCKET':
            offset = 0.0
            for obj in self.detections:
                if obj[0] == NAME_DIR['PACKET']:
                    offset = (obj[2] - FRAME_WIDTH / 2) / (FRAME_WIDTH / 2) + RATIO_OFFSET
                    break

            speed = DEFAULT_SPEED #if self.result['state'] in ('ok', 'road_ok') else 0.0 ## drivable area 없으면 stop
            self.return_val[:] = offset, speed, 'normal'
        elif self.state == 'LOADING':
            speed = 0.0
            self.return_val[:] = self.result['offset'], speed, 'normal'

            ## loading 끝난거 감지
            self._traverse(self.class_counters.get('BIG_PACKET', 0) == 0, 'EOR_RIGHT')
        else:
            self.return_val[:] = 0.0, 0.0, 'normal'
            raise Exception(f"[Decision] Unknown state: {self.state}")

    def run(self):
        self.running = True
        logger.info("[Decision] started")
        self.lane.pause()
        while self.running:
            t0 = time.perf_counter()
            self._process_yolo()
            self._process_lane()
            self._send(self.return_val[0], self.return_val[1], type=self.return_val[2])
            
            self._log_counter += 1
            if self._log_counter >= DEBUG_LOG_INTERVAL:
                self._log_counter = 0
                self._debug_log()

            sleep_time = self.interval - (time.perf_counter() - t0)
            if sleep_time > 0: time.sleep(sleep_time)



    def _send(self, offset_ratio: float, speed: float, type='normal'):
        if type == 'normal':
            L, R = self._compute_lr(offset_ratio, speed)
            self._base.send_command({"T": 1, "L": -R, "R": -L})
        elif type == 'spin':
            global INNER_SPEED
            temp = INNER_SPEED
            INNER_SPEED = 0.0
            L, R = self._compute_lr(offset_ratio, speed)
            self._base.send_command({"T": 1, "L": -R, "R": -L})
            INNER_SPEED = temp


    def _debug_log(self):
        offset     = self.result.get('offset', float('nan'))
        drv_state  = self.result.get('state', 'N/A')
        mask_ratio = self.result.get('mask_ratio', float('nan'))

        logger.debug(
            "[Decision] state=%-14s | drv=%-12s | offset=%+.3f | mask=%.3f | trav=%d | L=%+.3f R=%+.3f",
            self.state, drv_state, offset, mask_ratio, self.travere_counter,
            self.return_val[0], self.return_val[1]
        )
        if MORE:
            logger.debug("[Decision] class_counters=%s", self.class_counters)
            logger.debug("[Decision] detections=%s", self.detections)