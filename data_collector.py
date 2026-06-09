import cv2
import os
import time
import threading
from datetime import datetime
from flask import Flask, Response, jsonify

# ================================================================
# Constants - Camera / Video
# ================================================================
CAPTURE_W = 1280
CAPTURE_H = 720
DISPLAY_W = 1280
DISPLAY_H = 720
FRAMERATE = 30
FOURCC    = cv2.VideoWriter_fourcc(*'mp4v')

today    = datetime.now().strftime("%Y%m%d.%H%M")
BASE_DIR = f"dataset/raw/{today}"
os.makedirs(BASE_DIR, exist_ok=True)

VIDEO_LEFT  = os.path.join(BASE_DIR, "left.mp4")
VIDEO_RIGHT = os.path.join(BASE_DIR, "right.mp4")

# ================================================================
# GStreamer pipeline
# ================================================================
def gstreamer_pipeline(sensor_id=0, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, CAPTURE_W, CAPTURE_H, FRAMERATE, flip_method, DISPLAY_W, DISPLAY_H)
    )

# ================================================================
# Shared state
# ================================================================
running      = True
stream_on    = True
latest_left  = None
latest_right = None
frame_lock   = threading.Lock()

# ================================================================
# Flask
# ================================================================
app = Flask(__name__)

def generate(side):
    while running:
        if not stream_on:
            time.sleep(0.1)
            continue

        with frame_lock:
            fl = latest_left
            fr = latest_right

        if side == 'left' and fl is not None:
            frame = fl
        elif side == 'right' and fr is not None:
            frame = fr
        elif side == 'both' and fl is not None and fr is not None:
            frame = cv2.hconcat([fl, fr])
        else:
            time.sleep(0.01)
            continue

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
  <title>Dual Camera</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; }
    img  { max-width:100%; border:2px solid #333; }
    button {
      margin-top:16px; padding:12px 40px;
      font-size:18px; border:none; border-radius:8px;
      cursor:pointer; background:#2a2; color:#fff;
    }
    button.off { background:#a22; }
  </style>
</head>
<body>
  <h2>Dual Camera Stream</h2>
  <img id="feed" src="/stream">
  <br>
  <button id="btn" onclick="toggle()">⏹ Stream OFF</button>
  <script>
    let on = true;
    function toggle() {
      on = !on;
      fetch('/toggle', {method:'POST'});
      const btn = document.getElementById('btn');
      const feed = document.getElementById('feed');
      if (on) {
        feed.src = '/stream?' + Date.now();
        btn.textContent = '⏹ Stream OFF';
        btn.classList.remove('off');
      } else {
        feed.src = '';
        btn.textContent = '▶ Stream ON';
        btn.classList.add('off');
      }
    }
  </script>
</body>
</html>'''

@app.route('/toggle', methods=['POST'])
def toggle():
    global stream_on
    stream_on = not stream_on
    return jsonify({'stream': stream_on})

@app.route('/stream')
def stream():
    return Response(generate('both'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/left')
def stream_left():
    return Response(generate('left'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/right')
def stream_right():
    return Response(generate('right'), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# ================================================================
# Main
# ================================================================
def main():
    global running, latest_left, latest_right

    print("CSI 카메라 초기화 중...")
    cap_l = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    cap_r = cv2.VideoCapture(gstreamer_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    writer_l = cv2.VideoWriter(VIDEO_LEFT,  FOURCC, FRAMERATE, (DISPLAY_W, DISPLAY_H))
    writer_r = cv2.VideoWriter(VIDEO_RIGHT, FOURCC, FRAMERATE, (DISPLAY_W, DISPLAY_H))

    if not writer_l.isOpened() or not writer_r.isOpened():
        print("[ERROR] VideoWriter를 열 수 없습니다.")
        cap_l.release(); cap_r.release()
        return

    print(f"저장 경로: {BASE_DIR}")
    print("브라우저: http://<Jetson IP>:5000/")
    print("종료: Ctrl+C")
    print("=" * 50)

    threading.Thread(target=run_flask, daemon=True).start()

    frame_count = 0

    try:
        while True:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r:
                print("[WARN] 프레임 읽기 실패 — 재시도")
                continue

            writer_l.write(frame_l)
            writer_r.write(frame_r)

            with frame_lock:
                latest_left  = frame_l
                latest_right = frame_r

            frame_count += 1
            if frame_count % FRAMERATE == 0:
                print(f"\r[CAM] {frame_count // FRAMERATE}초 기록 중...", end='', flush=True)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료")

    finally:
        writer_l.release()
        writer_r.release()
        cap_l.release()
        cap_r.release()
        duration = frame_count / FRAMERATE
        print(f"[DONE] {frame_count}프레임 ({duration:.1f}초) 저장")
        print(f"  → {VIDEO_LEFT}")
        print(f"  → {VIDEO_RIGHT}")

if __name__ == "__main__":
    main()