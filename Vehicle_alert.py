from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify\
\pard\pardeftab720\partightenfactor0
\cf2 import cv2\
import torch\
import pyttsx3\
import numpy as np\
import time\
import threading_\
import warnings\
import datetime\
import os\
import base64\
from io import BytesIO\
from PIL import Image\
from collections import defaultdict\
\
warnings.filterwarnings("ignore", category=FutureWarning)\
\
app = Flask(__name__)\
app.secret_key = 'your_secret_key'\
\
# User credentials (use a DB in production)\
users = \{"admin": "password123"\}\
\
# User settings and data\
user_settings = \{\}\
detection_history = defaultdict(list)\
alert_logs = defaultdict(list)\
statistics = defaultdict(lambda: \{"detections": 0, "close_calls": 0, "total_distance": 0, "count": 0\})\
\
# Camera sources dictionary\
camera_sources = \{\}\
\
# Mobile camera streams dictionary\
mobile_frames = \{\}\
\
# Default settings\
DEFAULT_SETTINGS = \{\
    "audio_enabled": True,\
    "distance_threshold": 5.0,  # meters\
    "close_call_threshold": 2.0,  # meters\
    "alert_interval": 5,  # seconds\
    "night_mode": False,\
    "lane_departure_warning": False,\
    "camera_source": "mobile"  # Default to mobile camera\
\}\
\
# Create directories for storing images\
if not os.path.exists('static/captures'):\
    os.makedirs('static/captures')\
\
# Text-to-speech function (threaded)\
def speak(message):\
    def run():\
        try:\
            engine = pyttsx3.init()\
            engine.say(message)\
            engine.runAndWait()\
        except Exception as e:\
            print(f"Speech error: \{e\}")\
    threading.Thread(target=run, daemon=True).start()\
\
# Load YOLOv5 model\
try:\
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)\
    print("\'e2\'9c\'85 YOLOv5 model loaded successfully.")\
except Exception as e:\
    print(f"\'e2\uc0\u157 \'8c Error loading YOLOv5 model: \{e\}")\
    # Fallback to a simple placeholder if model fails to load\
    class DummyModel:\
        def __init__(self):\
            self.names = \{0: 'person', 1: 'car', 2: 'truck'\}\
        def __call__(self, frame):\
            class Results:\
                def __init__(self):\
                    self.xyxy = [np.array([])]\
            return Results()\
    model = DummyModel()\
\
# Constants\
TARGET_CLASSES = ["car", "motorcycle", "bicycle", "bus", "truck", "person"]\
FOCAL_LENGTH = 500\
KNOWN_WIDTH = 1.8\
CONFIDENCE_THRESHOLD = 0.5\
\
def calculate_distance(perceived_width):\
    return (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width if perceived_width > 0 else -1\
\
def enhance_night_image(frame):\
    try:\
        # Convert to grayscale\
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\
        # Apply histogram equalization\
        enhanced = cv2.equalizeHist(gray)\
        # Convert back to BGR\
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)\
        return enhanced\
    except Exception as e:\
        print(f"Night mode error: \{e\}")\
        return frame\
\
def get_camera_source(source_str):\
    """Parse camera source string to appropriate OpenCV source"""\
    try:\
        # Check if source is for mobile camera\
        if source_str == "mobile":\
            return "mobile"\
        # Check if source is a digit (local camera)\
        if source_str.isdigit():\
            return int(source_str)\
        # Otherwise assume it's an IP camera URL\
        return source_str\
    except:\
        # Default to mobile camera if there's any issue\
        return "mobile"\
\
def process_frame(frame, username):\
    """Process a frame for object detection and return annotated frame"""\
    try:\
        frame = cv2.resize(frame, (640, 480))\
        \
        # Apply night mode enhancement if enabled\
        if user_settings[username]["night_mode"]:\
            frame = enhance_night_image(frame)\
        \
        height, width = frame.shape[:2]\
        \
        # Process frame in regions (center and right side for overtaking detection)\
        right_roi = frame[:, width // 2:]\
        \
        try:\
            results = model(right_roi)\
            detections = results.xyxy[0].cpu().numpy()\
        except Exception as e:\
            print(f"Detection error: \{e\}")\
            detections = []\
        \
        current_time = time.time()\
        alerts = []\
        close_calls = 0\
        \
        # Process detections\
        for detection in detections:\
            try:\
                if len(detection) >= 6:  # Make sure detection has enough elements\
                    x1, y1, x2, y2, confidence, class_id = detection\
                    class_id = int(class_id)\
                    \
                    if confidence > CONFIDENCE_THRESHOLD and class_id < len(model.names) and model.names[class_id] in TARGET_CLASSES:\
                        x1 += width // 2  # Adjust coordinates for right ROI\
                        x2 += width // 2\
                        perceived_width = x2 - x1\
                        distance = calculate_distance(perceived_width)\
                        \
                        # Draw bounding box\
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)\
                        cv2.putText(frame, f"\{model.names[class_id]\}: \{distance:.2f\}m", \
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)\
                        \
                        if distance > 0 and distance < user_settings[username]["distance_threshold"]:\
                            alerts.append(f"\{model.names[class_id]\} detected on right")\
                        \
                        if distance > 0 and distance < user_settings[username]["close_call_threshold"]:\
                            close_calls += 1\
                            \
                            # Capture close call image\
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")\
                            filename = f"static/captures/\{username\}_\{timestamp\}.jpg"\
                            try:\
                                cv2.imwrite(filename, frame)\
                                alert_logs[username].append(\{\
                                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),\
                                    "type": model.names[class_id],\
                                    "distance": f"\{distance:.2f\}m",\
                                    "image": filename.replace("static/", "")\
                                \})\
                            except Exception as e:\
                                print(f"Error saving image: \{e\}")\
            except Exception as e:\
                print(f"Error processing detection: \{e\}")\
                continue\
        \
        # Add info overlay\
        cv2.putText(frame, f"Mode: \{'Night' if user_settings[username]['night_mode'] else 'Day'\}", \
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\
        cv2.putText(frame, f"Alert Distance: \{user_settings[username]['distance_threshold']\}m", \
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\
        cv2.putText(frame, f"Audio: \{'ON' if user_settings[username]['audio_enabled'] else 'OFF'\}", \
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\
        cv2.putText(frame, f"Camera: \{user_settings[username]['camera_source']\}", \
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\
                    \
        return frame, alerts, close_calls\
                \
    except Exception as e:\
        print(f"Frame processing error: \{e\}")\
        # Return an error frame\
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)\
        cv2.putText(error_frame, "Processing Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)\
        return error_frame, [], 0\
\
def generate_frames(username):\
    if username not in user_settings:\
        user_settings[username] = DEFAULT_SETTINGS.copy()\
    \
    try:\
        # Get camera source from user settings\
        source = get_camera_source(user_settings[username]["camera_source"])\
        \
        # For mobile camera, check if we have frames available\
        if source == "mobile":\
            while True:\
                if username in mobile_frames and mobile_frames[username] is not None:\
                    frame = mobile_frames[username]\
                    \
                    processed_frame, alerts, close_calls = process_frame(frame, username)\
                    \
                    # Handle alerts\
                    current_time = time.time()\
                    last_alert_time = getattr(generate_frames, 'last_alert_time', 0)\
                    \
                    if alerts and (current_time - last_alert_time > user_settings[username]["alert_interval"]):\
                        # Combine similar alerts\
                        unique_alerts = list(set(alerts))\
                        alert_message = " and ".join(unique_alerts)\
                        \
                        if user_settings[username]["audio_enabled"]:\
                            speak(alert_message)\
                        \
                        generate_frames.last_alert_time = current_time\
                        \
                        # Update statistics\
                        statistics[username]["detections"] += len(unique_alerts)\
                        statistics[username]["close_calls"] += close_calls\
                    \
                    # Convert frame to bytes for streaming\
                    ret, buffer = cv2.imencode('.jpg', processed_frame)\
                    frame_bytes = buffer.tobytes()\
                    yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
                else:\
                    # No mobile frames available yet\
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)\
                    cv2.putText(waiting_frame, "Waiting for mobile camera...", (160, 240), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)\
                    ret, buffer = cv2.imencode('.jpg', waiting_frame)\
                    frame_bytes = buffer.tobytes()\
                    yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
                \
                # Add a small delay\
                time.sleep(0.1)\
        else:\
            # Initialize camera for non-mobile sources\
            cap = cv2.VideoCapture(source)\
            \
            # Set timeout for IP camera connections\
            if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):\
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout\
            \
            if not cap.isOpened():\
                print(f"\'e2\uc0\u157 \'8c Error: Could not access camera source: \{source\}")\
                # Return a static error frame\
                while True:\
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)\
                    cv2.putText(frame, f"Camera Error: \{source\}", (150, 240), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\
                    ret, buffer = cv2.imencode('.jpg', frame)\
                    frame_bytes = buffer.tobytes()\
                    yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
                    time.sleep(1)\
        \
            print(f"\'e2\'9c\'85 Camera initialized successfully for user: \{username\}, source: \{source\}")\
            last_alert_time = 0\
            \
            while True:\
                success, frame = cap.read()\
                if not success:\
                    print(f"Failed to read frame from source: \{source\}")\
                    # Try to reconnect for IP cameras\
                    if isinstance(source, str) and (source.startswith('http') or source.startswith('rtsp')):\
                        print("Attempting to reconnect to IP camera...")\
                        cap.release()\
                        time.sleep(2)\
                        cap = cv2.VideoCapture(source)\
                    \
                    # Return an error frame while attempting to reconnect\
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)\
                    cv2.putText(error_frame, "Camera Connection Lost", (180, 240), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\
                    cv2.putText(error_frame, "Attempting to reconnect...", (170, 270), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\
                    ret, buffer = cv2.imencode('.jpg', error_frame)\
                    frame_bytes = buffer.tobytes()\
                    yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
                    continue\
\
                processed_frame, alerts, close_calls = process_frame(frame, username)\
                \
                # Handle alerts\
                current_time = time.time()\
                \
                if alerts and (current_time - last_alert_time > user_settings[username]["alert_interval"]):\
                    # Combine similar alerts\
                    unique_alerts = list(set(alerts))\
                    alert_message = " and ".join(unique_alerts)\
                    \
                    if user_settings[username]["audio_enabled"]:\
                        speak(alert_message)\
                    \
                    last_alert_time = current_time\
                    \
                    # Update statistics\
                    statistics[username]["detections"] += len(unique_alerts)\
                    statistics[username]["close_calls"] += close_calls\
                \
                # Convert frame to bytes for streaming\
                ret, buffer = cv2.imencode('.jpg', processed_frame)\
                frame_bytes = buffer.tobytes()\
                yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
    \
    except Exception as e:\
        print(f"Stream error: \{e\}")\
        # Return an error frame indefinitely\
        while True:\
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)\
            cv2.putText(error_frame, f"Stream Error: \{str(e)[:30]\}", (150, 240), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)\
            ret, buffer = cv2.imencode('.jpg', error_frame)\
            frame_bytes = buffer.tobytes()\
            yield (b'--frame\\r\\n' b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame_bytes + b'\\r\\n')\
            time.sleep(1)\
\
@app.route('/')\
def index():\
    if 'user' in session:\
        username = session['user']\
        if username not in user_settings:\
            user_settings[username] = DEFAULT_SETTINGS.copy()\
        return render_template('index.html', \
                               settings=user_settings[username],\
                               username=username)\
    return redirect(url_for('login'))\
\
@app.route('/login', methods=['GET', 'POST'])\
def login():\
    error = None\
    if request.method == 'POST':\
        username = request.form['username']\
        password = request.form['password']\
        if users.get(username) == password:\
            session['user'] = username\
            if username not in user_settings:\
                user_settings[username] = DEFAULT_SETTINGS.copy()\
            return redirect(url_for('index'))\
        error = "Invalid credentials."\
    return render_template('login.html', error=error)\
\
@app.route('/signup', methods=['GET', 'POST'])\
def signup():\
    error = None\
    if request.method == 'POST':\
        username = request.form['username']\
        password = request.form['password']\
        if username in users:\
            error = "Username already exists."\
        else:\
            users[username] = password\
            user_settings[username] = DEFAULT_SETTINGS.copy()\
            return redirect(url_for('login'))\
    return render_template('signup.html', error=error)\
\
@app.route('/logout')\
def logout():\
    session.clear()\
    return redirect(url_for('login'))\
\
@app.route('/toggle_audio')\
def toggle_audio():\
    if 'user' in session:\
        username = session['user']\
        user_settings[username]["audio_enabled"] = not user_settings[username]["audio_enabled"]\
    return redirect(url_for('index'))\
\
@app.route('/toggle_night_mode')\
def toggle_night_mode():\
    if 'user' in session:\
        username = session['user']\
        user_settings[username]["night_mode"] = not user_settings[username]["night_mode"]\
    return redirect(url_for('index'))\
\
@app.route('/update_settings', methods=['POST'])\
def update_settings():\
    if 'user' in session:\
        username = session['user']\
        try:\
            data = request.json\
            \
            # Update settings\
            if "distance_threshold" in data:\
                user_settings[username]["distance_threshold"] = float(data["distance_threshold"])\
            if "close_call_threshold" in data:\
                user_settings[username]["close_call_threshold"] = float(data["close_call_threshold"])\
            if "alert_interval" in data:\
                user_settings[username]["alert_interval"] = int(data["alert_interval"])\
            if "camera_source" in data:\
                # Update camera source\
                user_settings[username]["camera_source"] = data["camera_source"]\
            if "lane_departure_warning" in data:\
                user_settings[username]["lane_departure_warning"] = bool(data["lane_departure_warning"])\
            if "night_mode" in data:\
                user_settings[username]["night_mode"] = bool(data["night_mode"])\
                \
            return jsonify(\{"status": "success"\})\
        except Exception as e:\
            return jsonify(\{"status": "error", "message": str(e)\})\
    return jsonify(\{"status": "error", "message": "Not logged in"\})\
\
@app.route('/mobile_frame', methods=['POST'])\
def mobile_frame():\
    """Endpoint to receive frames from mobile camera"""\
    if 'user' in session:\
        username = session['user']\
        try:\
            data = request.json\
            if 'image_data' in data:\
                # Decode base64 image\
                image_data = data['image_data'].split(',')[1] if ',' in data['image_data'] else data['image_data']\
                img_bytes = base64.b64decode(image_data)\
                img = Image.open(BytesIO(img_bytes))\
                \
                # Convert to OpenCV format\
                frame = np.array(img)\
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)\
                \
                # Store frame for this user\
                mobile_frames[username] = frame\
                \
                return jsonify(\{"status": "success"\})\
            return jsonify(\{"status": "error", "message": "No image data"\})\
        except Exception as e:\
            print(f"Error processing mobile frame: \{e\}")\
            return jsonify(\{"status": "error", "message": str(e)\})\
    return jsonify(\{"status": "error", "message": "Not logged in"\})\
\
@app.route('/camera_settings')\
def camera_settings():\
    if 'user' in session:\
        username = session['user']\
        return render_template('camera_settings.html', \
                              settings=user_settings[username],\
                              username=username)\
    return redirect(url_for('login'))\
\
@app.route('/dashboard')\
def dashboard():\
    if 'user' in session:\
        username = session['user']\
        \
        # Calculate average distance\
        avg_distance = 0\
        if statistics[username]["count"] > 0:\
            avg_distance = statistics[username]["total_distance"] / statistics[username]["count"]\
        \
        return render_template('dashboard.html', \
                               stats=statistics[username],\
                               avg_distance=avg_distance,\
                               alerts=alert_logs[username],\
                               username=username)\
    return redirect(url_for('login'))\
\
@app.route('/video_feed')\
def video_feed():\
    if 'user' in session:\
        return Response(generate_frames(session['user']),\
                        mimetype='multipart/x-mixed-replace; boundary=frame')\
    return redirect(url_for('login'))\
\
@app.route('/clear_logs')\
def clear_logs():\
    if 'user' in session:\
        username = session['user']\
        alert_logs[username] = []\
    return redirect(url_for('dashboard'))\
\
@app.route('/reset_stats')\
def reset_stats():\
    if 'user' in session:\
        username = session['user']\
        statistics[username] = \{"detections": 0, "close_calls": 0, "total_distance": 0, "count": 0\}\
    return redirect(url_for('dashboard'))\
\
if __name__ == '__main__':\
    # Running with host='0.0.0.0' makes the server accessible on your local network\
    app.run(host='0.0.0.0', port=5000, debug=True)}
