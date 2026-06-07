# main.py
import time
import logging
from logging.handlers import RotatingFileHandler
import pycuda.driver as cuda

from config import (
    CAMERA_ID, FPS, FRAME_WIDTH, FRAME_HEIGHT,
    LANE_ENGINE_PATH, CONF_THR, Y_RATIO, RATIO_OFFSET,
    YOLO_SIGNAL_PATH, YOLO_SIGNAL_CONF_THR, YOLO_SIGNAL_N_FRAME,
    DRIVABLE_ENGINE_PATH, YOLO_DRIVABLE_CONF_THR, YOLO_DRIVABLE_N_FRAME,
    DECISION_INTERVAL,
)
from camera_thread import CameraThread
from lane_thread import LaneThread
from yolo_signal_thread import YoloSignalThread
from yolo_drivable_thread import YoloDrivableThread
from decision_thread import DecisionThread

LOG_FILE   = "round2.log"
ENABLE_LOG = True

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

decision_logger = logging.getLogger("decision_thread")
decision_logger.setLevel(logging.DEBUG)

if ENABLE_LOG:
    _fh = RotatingFileHandler(LOG_FILE, mode='w', maxBytes=5*1024*1024,
                              backupCount=5, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    decision_logger.addHandler(_fh)


def main():
    cuda.init()

    cam           = CameraThread(sensor_id=CAMERA_ID, fps=FPS,
                                 width=FRAME_WIDTH, height=FRAME_HEIGHT)
    lane          = LaneThread(camera=cam, engine_path=LANE_ENGINE_PATH,
                               conf_thr=CONF_THR, y_ratio=Y_RATIO,
                               ratio_offset=RATIO_OFFSET)
    yolo_signal   = YoloSignalThread(camera=cam, engine_path=YOLO_SIGNAL_PATH,
                                     conf_thr=YOLO_SIGNAL_CONF_THR,
                                     n_frame=YOLO_SIGNAL_N_FRAME)
    yolo_drivable = YoloDrivableThread(camera=cam, engine_path=DRIVABLE_ENGINE_PATH,
                                   conf=YOLO_DRIVABLE_CONF_THR,
                                   n_frame=YOLO_DRIVABLE_N_FRAME)
    decision      = DecisionThread(lane=lane, yolo=yolo_signal,
                                   drivable=yolo_drivable,
                                   interval=DECISION_INTERVAL)

    cam.start()
    lane.start()
    yolo_signal.start()
    yolo_drivable.start()
    decision.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        decision.stop()
        yolo_drivable.stop()
        lane.stop()
        yolo_signal.stop()
        cam.stop()
        if ENABLE_LOG:
            _fh.close()
        logging.info("All threads stopped")


if __name__ == "__main__":
    main()