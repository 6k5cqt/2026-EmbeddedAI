# control_thread.py
import threading
import time
import logging

from ctrl_utils import BaseController

logger = logging.getLogger(__name__)

UART_DEV  = '/dev/ttyUSB0'
BAUD_RATE = 115200

Kp           = 0.6
Ki           = 0.0
Kd           = 0.00
I_LIMIT      = 0.3
INTENSITY_MIN = 0.25
INTENSITY_MAX = 0.5
MARGIN_ERROR  = 0.1
INNER_SPEED   = 0.03


class ControlThread(threading.Thread):
    def __init__(self, interval: float):
        super().__init__(daemon=True, name="ControlThread")
        self.interval      = interval
        self.running       = False
        self._offset_ratio = 0.0
        self._speed        = 0.0
        self._lock         = threading.Lock()
        self._base         = BaseController(UART_DEV, BAUD_RATE)

        self._integral_sum = 0.0
        self._prev_ratio   = 0.0
        self._prev_time    = time.time()

    def set_command(self, offset_ratio: float, speed: float):
        with self._lock:
            self._offset_ratio = offset_ratio
            self._speed        = speed

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

        p_term = Kp * abs(offset_ratio)
        d_term = Kd * abs(derivative)

        total = p_term
        if offset_ratio * derivative < 0:
            total -= d_term
        else:
            total += d_term

        intensity = max(INTENSITY_MIN, min(INTENSITY_MAX, total))

        if offset_ratio > 0:
            return intensity, INNER_SPEED
        else:
            return INNER_SPEED, intensity

    def run(self):
        self.running = True
        logger.info("[Control] started")

        while self.running:
            t0 = time.perf_counter()

            with self._lock:
                offset_ratio = self._offset_ratio
                speed        = self._speed

            L, R = self._compute_lr(offset_ratio, speed)
            self._base.send_command({"T": 1, "L": -R, "R": -L})
            logger.debug(f"[Control] offset={offset_ratio:+.2f} L={L:.2f} R={R:.2f}")

            elapsed = time.perf_counter() - t0
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        try:
            self._base.send_command({"T": 1, "L": 0.0, "R": 0.0})
        except Exception:
            pass
        self.join()
        logger.info("[Control] stopped")