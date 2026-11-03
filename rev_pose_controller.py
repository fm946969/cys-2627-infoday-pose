import cv2
import mediapipe as mp
import serial
import time
import numpy as np # NumPy is needed for image operations

# =====================================================================
# 🛠️ 1. System Parameters & Logo Setup
# =====================================================================
# --- Serial Port (Modify to your micro:bit's COM Port, Please check Windows's "Device Manager") ---
SERIAL_PORT = 'COM5'  
BAUD_RATE = 115200

# --- Logo Setup ---
LOGO_FILE = '3cys-schoolLogo.png'
LOGO_SIZE = 120  # You can change the size of the logo here (e.g., 100, 120, 150)
LOGO_MARGIN = 15 # Margin from the top-left corner

# Load the logo from the file
try:
    logo = cv2.imread(LOGO_FILE)
    if logo is None:
        raise FileNotFoundError
    # Resize the logo to our desired size
    logo = cv2.resize(logo, (LOGO_SIZE, LOGO_SIZE))
    print(f"✅ Successfully loaded and resized school logo '{LOGO_FILE}'.")
except FileNotFoundError:
    print(f"⚠️ Warning: Logo file '{LOGO_FILE}' not found. The script will run without the logo.")
    logo = None
except Exception as e:
    print(f"⚠️ Warning: Could not process logo. Error: {e}. The script will run without the logo.")
    logo = None

# --- Serial Connection ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"✅ Successfully connected to micro:bit (Port: {SERIAL_PORT})")
except Exception as e:
    print(f"⚠️ Could not connect to serial port {SERIAL_PORT}. Running in 'No-Hardware Test Mode'.")
    ser = None

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Drawing Styles ---
GREEN_COLOR = (0, 255, 0)
draw_spec_landmarks = mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=3, circle_radius=3)
draw_spec_connections = mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=2)

# =====================================================================
# 🏃 2. Pose-to-Command Logic (No changes here)
# =====================================================================
def get_control_command(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return 'F', "FORWARD (↑)"
    elif left_wrist.y > left_hip.y and right_wrist.y > right_hip.y:
        return 'B', "BACKWARD (↓)"
    elif abs(right_wrist.x - right_shoulder.x) > 0.25 and right_wrist.y < right_hip.y and left_wrist.y > left_shoulder.y:
        return 'L', "TURN LEFT (←)"
    elif abs(left_wrist.x - left_shoulder.x) > 0.25 and left_wrist.y < left_hip.y and right_wrist.y > right_shoulder.y:
        return 'R', "TURN RIGHT (→)"
    else:
        return 'S', "STOP (🛑)"

# =====================================================================
# 📷 3. Main Loop for Video Capture and Processing
# =====================================================================
cap = cv2.VideoCapture(0)
last_cmd = ""
last_send_time = time.time()
SEND_INTERVAL = 0.15

print("\n🚀 AI Pose Control System Activated! Please stand in front of the camera.")
print("💡 Tip: Press the 'q' key on the video window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_cmd = 'S'
    action_text = "STOP (🛑)"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw_spec_landmarks,
            connection_drawing_spec=draw_spec_connections)
        current_cmd, action_text = get_control_command(results.pose_landmarks.landmark)

    # =====================================================================
    # ✨ NEW: Overlay the school logo onto the frame
    # =====================================================================
    if logo is not None:
        # Define the Region of Interest (ROI) in the top-left corner
        roi = frame[LOGO_MARGIN:LOGO_SIZE+LOGO_MARGIN, LOGO_MARGIN:LOGO_SIZE+LOGO_MARGIN]
        # Place the logo onto the frame
        roi[:] = logo

    # =====================================================================
    # 📡 4. Send Command via USB Serial (No changes here)
    # =====================================================================
    now = time.time()
    if (current_cmd != last_cmd) or (now - last_send_time > SEND_INTERVAL):
        if ser:
            try:
                ser.write(f"{current_cmd}\n".encode('utf-8'))
            except Exception as e:
                print(f"⚠️ Error sending data to serial port: {e}")
        last_cmd = current_cmd
        last_send_time = now

    # =====================================================================
    # 📺 5. Render the On-Screen Display (HUD) - MOVED TO TOP-RIGHT
    # =====================================================================
    # Draw a semi-transparent background for the text
    cv2.rectangle(frame, (w - 340, 10), (w - 10, 95), (0, 0, 0), -1)
    # Display the current command and action
    cv2.putText(frame, f"USB Sent: {current_cmd}", (w - 330, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Action  : {action_text}", (w - 330, 78), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the final image
    cv2.imshow('CYS AI Pose Controller (Maqueen Plus V2)', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# =====================================================================
# 🔌 6. Cleanup (No changes here)
# =====================================================================
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    print("🔌 Serial port closed safely. System shutting down.")
