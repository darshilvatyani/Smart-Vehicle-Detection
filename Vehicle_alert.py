from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import torch
import pyttsx3
import numpy as np
import time
import threading
import warnings
import datetime
import os
import base64
from io import BytesIO
from PIL import Image
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.secret_key = "your_secret_key"

users = {"admin": "password123"}

user_settings = {}
detection_history = defaultdict(list)
alert_logs = defaultdict(list)
statistics = defaultdict(lambda: {
    "detections": 0,
    "close_calls": 0,
    "total_distance": 0,
    "count": 0
})

camera_sources = {}
mobile_frames = {}

DEFAULT_SETTINGS = {
    "audio_enabled": True,
    "distance_threshold": 5.0,
    "close_call_threshold": 2.0,
    "alert_interval": 5,
    "night_mode": False,
    "lane_departure_warning": False,
    "camera_source": "mobile"
}

os.makedirs("static/captures", exist_ok=True)

def speak(message):
    def run():
        try:
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(e)
    threading.Thread(target=run, daemon=True).start()

try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
except Exception:
    class DummyModel:
        def __init__(self):
            self.names = {0: "person", 1: "car", 2: "truck"}
        def __call__(self, frame):
            class Results:
                def __init__(self):
                    self.xyxy = [np.array([])]
            return Results()
    model = DummyModel()

TARGET_CLASSES = ["car", "motorcycle", "bicycle", "bus", "truck", "person"]
FOCAL_LENGTH = 500
KNOWN_WIDTH = 1.8
CONFIDENCE_THRESHOLD = 0.5

def calculate_distance(width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / width if width > 0 else -1

def enhance_night_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def get_camera_source(src):
    if src == "mobile":
        return "mobile"
    if src.isdigit():
        return int(src)
    return src

def process_frame(frame, username):
    frame = cv2.resize(frame, (640, 480))

    if user_settings[username]["night_mode"]:
        frame = enhance_night_image(frame)

    h, w = frame.shape[:2]
    roi = frame[:, w // 2:]

    try:
        results = model(roi)
        detections = results.xyxy[0].cpu().numpy()
    except:
        detections = []

    alerts = []
    close_calls = 0

    for det in detections:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, conf, cid = det
        cid = int(cid)

        if conf < CONFIDENCE_THRESHOLD or cid >= len(model.names):
            continue

        label = model.names[cid]
        if label not in TARGET_CLASSES:
            continue

        x1 += w // 2
        x2 += w // 2

        distance = calculate_distance(x2 - x1)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {distance:.2f}m",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if 0 < distance < user_settings[username]["distance_threshold"]:
            alerts.append(f"{label} detected")

        if 0 < distance < user_settings[username]["close_call_threshold"]:
            close_calls += 1
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"static/captures/{username}_{ts}.jpg"
            cv2.imwrite(path, frame)
            alert_logs[username].append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": label,
                "distance": f"{distance:.2f}m",
                "image": path.replace("static/", "")
            })

    return frame, alerts, close_calls

def generate_frames(username):
    if username not in user_settings:
        user_settings[username] = DEFAULT_SETTINGS.copy()

    source = get_camera_source(user_settings[username]["camera_source"])
    last_alert_time = 0

    if source == "mobile":
        while True:
            if username in mobile_frames:
                frame = mobile_frames[username]
                processed, alerts, close_calls = process_frame(frame, username)

                now = time.time()
                if alerts and now - last_alert_time > user_settings[username]["alert_interval"]:
                    if user_settings[username]["audio_enabled"]:
                        speak(" and ".join(set(alerts)))
                    last_alert_time = now

                _, buffer = cv2.imencode(".jpg", processed)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(0.1)

    cap = cv2.VideoCapture(source)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        processed, alerts, close_calls = process_frame(frame, username)

        now = time.time()
        if alerts and now - last_alert_time > user_settings[username]["alert_interval"]:
            if user_settings[username]["audio_enabled"]:
                speak(" and ".join(set(alerts)))
            last_alert_time = now

        _, buffer = cv2.imencode(".jpg", processed)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

@app.route("/")
def index():
    if "user" in session:
        user_settings.setdefault(session["user"], DEFAULT_SETTINGS.copy())
        return render_template("index.html", settings=user_settings[session["user"]])
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if users.get(request.form["username"]) == request.form["password"]:
            session["user"] = request.form["username"]
            return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/video_feed")
def video_feed():
    if "user" in session:
        return Response(generate_frames(session["user"]),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    return redirect(url_for("login"))

@app.route("/mobile_frame", methods=["POST"])
def mobile_frame():
    if "user" not in session:
        return jsonify({"status": "unauthorized"})

    data = request.json
    img_data = base64.b64decode(data["image_data"].split(",")[1])
    img = Image.open(BytesIO(img_data))
    mobile_frames[session["user"]] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

