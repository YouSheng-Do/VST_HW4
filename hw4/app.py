from flask import Flask, Response
import cv2

def capture_video():
    # 開啟攝像頭
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            return "Camera cannot capture video"
            break

        # 將frame轉換為JPEG格式
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Live Streaming Server"

@app.route('/')
def video_feed():
    return Response(capture_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
