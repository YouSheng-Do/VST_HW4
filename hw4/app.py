import cv2
import numpy as np

from api.deepsort import DeepSORTTracker
from api.yolo import YOLOPersonDetector
from flask import Flask, Response, render_template, request, jsonify

# constants
YOLO_MODEL = "./checkpoints/yolov7x.pt"
REID_MODEL = "./checkpoints/ReID.pb"
MAX_COS_DIST = 0.5
MAX_TRACK_AGE = 100

detector = YOLOPersonDetector()
detector.load(YOLO_MODEL)
tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)

select_coordinates = None

def capture_video():
    # 開啟攝像頭
    cap = cv2.VideoCapture(0)

    global select_coordinates
    last_coordinates = None
    i=0
    last_total_people = 0
    to_draw = []
    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            return "Camera cannot capture video"
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process YOLO detections
        detections = detector.detect(frame)
        try:
            bboxes, scores, _ = np.hsplit(detections, [4, 5])
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
            n_objects = detections.shape[0]
        except ValueError:
            bboxes = np.empty(0)
            scores = np.empty(0)
            n_objects = 0

        if n_objects > last_total_people:
            for i in range(n_objects-last_total_people):
                to_draw.append(True)

        to_draw, last_coordinates = tracker.track(frame, bboxes, scores.flatten(),to_draw, select_coordinates, last_coordinates)
        last_total_people = n_objects

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 將frame轉換為JPEG格式
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(capture_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def stream():
    return render_template('stream.html')  # 假設您的HTML檔案名為stream.html

@app.route('/handle_click', methods=['POST'])
def handle_click():
    global select_coordinates
    data = request.get_json()
    select_coordinates = (data['x'], data['y'])
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
