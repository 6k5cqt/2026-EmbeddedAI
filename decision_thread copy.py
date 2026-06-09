# decision_thread.py
import threading
import time
import logging

from ctrl_utils import BaseController

logger = logging.getLogger(__name__)

# ── Params ────────────────────────────────────────────────────
DEBUG_LOG_INTERVAL = 1
DEBUG_CONTROL = False

NAME_DIR = {
    'LEFT': 'sign_left', 'RIGHT': 'sign_right',
    'STOP': 'sign_stop', 'SLOW': 'sign_slow',
    'RED': 'sign_red_light', 'GREEN': 'sign_green_light', 'ROVER': 'rover'
}

# 각 class별 size가 얼마일 때 counter를 올릴지
CLASS_SIZE_TR = {
    'LEFT':  350,
    'RIGHT': 350,
    'STOP':  1000,
    'SLOW':  1000,
    'RED':   1100,
    'GREEN': 1000,
    'ROVER': 1500,
}

#각 class별 counter가 얼마일 때 traverse할지
COUNTER_TR = {
    'LEFT':  4,
    'RIGHT': 4,
    'STOP':  4,
    'SLOW':  3,
    'RED':   7,
    'GREEN': 5,
    'ROVER': 3,
    'DEFAULT':10,
}

# for rotary
IN_COUNT_TR      = 0.7
OUT_COUNT_TR     = 0.3
IN_COUNT_MARGIN  = 6
OUT_COUNT_MARGIN = 20

# for other signs
STOP_PERIOD   = 4
SLOW_PERIOD   = 5
WAIT_PERIOD   = 2
SLOW_SPEED    = 0.03
DEFAULT_SPEED = 0.2

# utils
DISPLAY_W     = 640
UART_DEV  = '/dev/ttyUSB0'
BAUD_RATE = 115200

# PID
Kp           = 0.8
Ki           = 0.0
Kd           = 0.2
I_LIMIT      = 0.3

INTENSITY_MIN = 0.3
INTENSITY_MAX = 0.5
MARGIN_ERROR  = 0.08
INNER_SPEED   = 0.05


class DecisionThread(threading.Thread):
    def __init__(self, lane, yolo, drivable, interval: float):
        super().__init__(daemon=True, name="DecisionThread")
        self.lane     = lane
        self.yolo     = yolo
        self.drivable = drivable
        self.interval = interval
        self.running  = False


        ##################### offroad
        self.speed = 0.0
        
    
        # common
        self.state            = 'OFFROAD'
        self.traverse_counter = 0
        self.return_val       = [0.0, 0.0]
        self._log_counter     = 0

        # rotary
        self.exit_count   = 0
        self.exit_target  = 0
        self.rotary_state = 'OUT'
        self.is_rover     = False

        # stop
        self.prev_time = 0.0

        # rover
        self.prev_pos = 640
        self.curr_pos = 0

        # yolo
        self.curr_class     = 'NO_OBJECT'
        self.curr_size      = 0.0
        self.detections     = []
        self.class_counters = {}

        # PID
        self._integral_sum = 0.0
        self._prev_ratio   = 0.0
        self._prev_time    = time.time()

        # UART
        self._base = BaseController(UART_DEV, BAUD_RATE)

    # ── PID ───────────────────────────────────────────────────
    def _compute_lr(self, offset_ratio: float, speed: float):
        if speed == 0:
            return 0.0, 0.0

        if abs(offset_ratio) < MARGIN_ERROR:
            self._prev_ratio   = 0.0
            self._integral_sum = 0.0
            return speed, speed

        now = time.time()
        dt  = now - self._prev_time
        self._prev_time = now

        derivative       = (offset_ratio - self._prev_ratio) / dt
        self._prev_ratio = offset_ratio

        self._integral_sum = max(-I_LIMIT, min(I_LIMIT, self._integral_sum + offset_ratio * dt))

        p     = Kp * abs(offset_ratio)
        i     = Ki * self._integral_sum
        d     = Kd * abs(derivative)
        total = p + i + (d if offset_ratio * derivative >= 0 else -d)

        intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, total))

        return (intensity, INNER_SPEED) if offset_ratio > 0 else (INNER_SPEED, intensity)

    

    # ── Traverse helpers ──────────────────────────────────────
    def _traverse(self, condition: bool, target_state: str):
        if condition:
            self.state = target_state
            self.class_counters = {}

    # ── YOLO 처리 ─────────────────────────────────────────────

    def _update_class_counters(self, detections):
        detected = {d[0]: d[1] for d in detections}  # {class_name: size}

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
        self._update_class_counters(detections) #detections에 있는 애들 중 만족하는 놈들 counter + or -

        if self.state == 'DEFAULT':
            # TR 넘은 애들 찾아서 traverse
            triggered = self._check_counter()
            if   triggered == 'LEFT':
                self.exit_target = 4
                self._traverse(True, 'ROTARY')
                if self.class_counters.get('ROVER', 0) > COUNTER_TR['ROVER']: #rotary 들어갈 때 ROVER도 카운터 만족하면 들어가기
                    self.is_rover = True
            elif triggered == 'RIGHT':
                self.exit_target = 1
                self._traverse(True, 'ROTARY')
                if self.class_counters.get('ROVER', 0) > COUNTER_TR['ROVER']:
                    self.is_rover = True
            elif triggered == 'STOP':  self.prev_time = time.time(); self.state = 'STOP'
            elif triggered == 'SLOW':  self.prev_time = time.time(); self.state = 'SLOW'
            elif triggered == 'RED':   self.state = 'SIGNAL'
            elif triggered == 'GREEN': self.state = 'SIGNAL_PASSING' #_traverse를 쓰면 카운터를 flush 해서 사용하면 안됨

    # ── Lane 처리 ─────────────────────────────────────────────
    def _process_lane(self, offset_ratios):
        if self.state == 'DEFAULT':
            self.return_val[:] = offset_ratios[-1], DEFAULT_SPEED

        elif self.state == 'ROTARY':
            if len(offset_ratios) >= 3:
                offset_ratios = [offset_ratios[0], offset_ratios[-1]]
            n   = len(offset_ratios)
            gap = offset_ratios[-1] - offset_ratios[0] if n >= 2 else 0.0

            if self.class_counters.get('ROVER', 0) > COUNTER_TR['ROVER']:
                self.is_rover = True

            if self.rotary_state == 'OUT':
                if n >= 2 and gap > IN_COUNT_TR:
                    self.traverse_counter += 1
                else:
                    self.traverse_counter = max(0, self.traverse_counter - 1)
                if self.traverse_counter > IN_COUNT_MARGIN:
                    self.exit_count += 1
                    self.traverse_counter = 0
                    self.rotary_state = 'IN'
            elif self.rotary_state == 'IN':
                if gap < OUT_COUNT_TR:
                    self.traverse_counter += 1
                else:
                    self.traverse_counter = max(0, self.traverse_counter - 1)
                if self.traverse_counter >= OUT_COUNT_MARGIN:
                    self.rotary_state = 'OUT'
                    self.traverse_counter = 0

            if self.is_rover and self.exit_count == 1:
                self.state    = 'WAIT'
                self.is_rover = False
                self.return_val[:] = offset_ratios[-1], 0.0
            else:
                if self.exit_count == self.exit_target:
                    self.state = 'DEFAULT'
                    self.class_counters = {}
                    self.rotary_state     = 'OUT'
                    self.exit_count       = 0
                    self.exit_target      = 0
                    self.traverse_counter = 0
                    self.return_val[:] = offset_ratios[-1], DEFAULT_SPEED
                elif self.exit_count > 1:
                    self.return_val[:] = offset_ratios[0], DEFAULT_SPEED
                else:
                    self.return_val[:] = offset_ratios[-1], DEFAULT_SPEED
        elif self.state == 'STOP':
            if time.time() - self.prev_time < SLOW_PERIOD:
                self.return_val[:] = 0.0, 0.0
                self.class_counters = {} #여기 있는 동안은 카운팅 하면 다시 0으로 돌아오는데 너무 오래걸림
            else:
                self.return_val[:] = offset_ratios[-1], DEFAULT_SPEED
                self._traverse(self.class_counters.get('STOP', 0) == 0, 'DEFAULT')
        elif self.state == 'SLOW':
            if time.time() - self.prev_time < STOP_PERIOD:
                self.return_val[:] = offset_ratios[-1], SLOW_SPEED
                self.class_counters = {}
            else:
                self.return_val[:] = offset_ratios[-1], SLOW_SPEED
                self._traverse(self.class_counters.get('SLOW', 0) == 0, 'DEFAULT')

        elif self.state == 'SIGNAL':
            self.return_val[:] = offset_ratios[-1], 0.0
            self._traverse(self.class_counters.get('GREEN', 0) >= COUNTER_TR['GREEN'], 'SIGNAL_PASSING')

        elif self.state == 'SIGNAL_PASSING':
            self.return_val[:] = 0.0, DEFAULT_SPEED
            self._traverse(
                self.class_counters.get('RED', 0) == 0 and self.class_counters.get('GREEN', 0) == 0, 'DEFAULT')

        elif self.state == 'WAIT':
            self.return_val[:] = offset_ratios[-1], 0.0
            for obj in self.detections:
                if obj[0] == NAME_DIR['ROVER']:
                    self.curr_pos = obj[2]
            if self.curr_pos > DISPLAY_W * (1/2):
                self._traverse(self.curr_pos > self.prev_pos, 'ROTARY')
                if self.state == 'ROTARY':
                    time.sleep(WAIT_PERIOD)
                self.prev_pos = self.curr_pos
                self.class_counters = {}
        # ── Here for Offroad !!!!!!!!!!!!!!  ─────────────────────────────────────────────
        #
        # - OFFROAD state에 들어오기 전에 self.lane.pause() 해주고, OFFROAD에서 나갈 때 self.lane.resume()
        # - 현재는 main loop 전에 pause하니까 지워야됨
        # - OFFROAD 에서는 self.speed를 사용하고 이는 OFFROAD 상태일 때만 _process_lane 전에 세팅해준다.
        #
        #
        elif self.state == 'OFFROAD':
            self.return_val[:] = offset_ratios[-1], self.speed
        else:
            raise Exception(f"[Decision] unknown state: {self.state}")

    # ── Main loop ─────────────────────────────────────────────
    def run(self):
        self.running = True
        logger.info("[Decision] started")

        self.lane.pause() ###################### 지워야됨!!

        while self.running:
            t0 = time.perf_counter()

            if self.state == 'OFFROAD':
                result = self.drivable.get_result()
                # offset ratio 계산
                offset_ratios = [result['offset']] if result['state'] == 'ok' else [0.0]
                # speed 세팅
                if result['state'] == 'stop' or result['state'] == 'no_detection':
                    self.speed = 0.0
                else:
                    self.speed = DEFAULT_SPEED
            else:
                offset_ratios = self.lane.get_result()
            detections    = self.yolo.get_result()

            if DEBUG_CONTROL:
                self._send(offset_ratios[-1], DEFAULT_SPEED)
            else:
                self._process_yolo(detections)
                self._process_lane(offset_ratios)
                self._send(self.return_val[0], self.return_val[1])

            elapsed = time.perf_counter() - t0
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Debug
    def _send(self, offset_ratio: float, speed: float):
        L, R = self._compute_lr(offset_ratio, speed)
        self._base.send_command({"T": 1, "L": -R, "R": -L})
        self._log_counter += 1
        if self._log_counter % DEBUG_LOG_INTERVAL == 0:
            base = (
                f"state={self.state} | "
                f"offset={offset_ratio:+.2f} | "
                f"L={L:.2f} R={R:.2f}"
            )
            if self.state == 'DEFAULT':
                extra = f" | counters={self.class_counters}"
            elif self.state in ('ROTARY', 'WAIT'):
                extra = (
                    f" | rotary={self.rotary_state}"
                    f" exit={self.exit_count}/{self.exit_target}"
                    f" traverse={self.traverse_counter}"
                )
                if self.state == 'WAIT':
                    extra += f" | rover_pos={self.curr_pos} prev={self.prev_pos}"
            elif self.state == 'STOP':
                extra = (
                    f" | remaining={max(0, STOP_PERIOD - (time.time() - self.prev_time)):.1f}s"
                    f" traverse={self.traverse_counter}"
                )
            elif self.state == 'SLOW':
                extra = (
                    f" | slow_counter={self.class_counters.get('SLOW', 0)}"
                    f" traverse={self.traverse_counter}"
                )
            elif self.state in ('SIGNAL', 'SIGNAL_PASSING'):
                extra = (
                    f" | RED={self.class_counters.get('RED', 0)}"
                    f" GREEN={self.class_counters.get('GREEN', 0)}"
                    f" traverse={self.traverse_counter}"
                )
            elif self.state == 'OFFROAD':
                r = self.drivable.get_result()
                extra = f" | drivable_state={r['state']} offset={r['offset']:+.3f} mask={r['mask_ratio']:.3f} speed={self.speed:.2f}"
            else:
                extra = f" | traverse={self.traverse_counter}"

            logger.debug(base + extra)
            logger.debug(f"  detections={self.detections}")