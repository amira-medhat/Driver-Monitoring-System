import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize Mediapipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize GUI
root = tk.Tk()
root.title("Head Movement and Gaze Detection")

video_label = Label(root)
video_label.pack()

gaze_label = Label(root, text="Gaze: ", font=("Arial", 14))
gaze_label.pack()

pitch_label = Label(root, text="Pitch: ", font=("Arial", 14))
pitch_label.pack()

yaw_label = Label(root, text="Yaw: ", font=("Arial", 14))
yaw_label.pack()

roll_label = Label(root, text="Roll: ", font=("Arial", 14))
roll_label.pack()

alert_label = Label(root, text="", font=("Arial", 16), fg="red")
alert_label.pack()

# Global variables
start_time = time.time()
threshold_time = 20
baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0
baseline_data = []
baseline_set = False

gaze_start_time = None
gaze_alert_triggered = False
gaze_abnormal_duration = 5  # Alert after 5 seconds

head_alert_start_time = None
head_alert_triggered = False
head_abnormal_duration = 5  # Alert after 5 seconds

PITCH_THRESHOLD = 10
YAW_THRESHOLD = 10
ROLL_THRESHOLD = 10
EAR_THRESHOLD = 0.35

cap = cv2.VideoCapture(0)


def calculate_angles(landmarks):
    """Calculate pitch, yaw, and roll from face landmarks correctly"""
    
    nose_tip = landmarks[1]
    chin = landmarks[152]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    forehead = landmarks[10]

    pitch = np.degrees(np.arctan2(chin.y - nose_tip.y, abs(chin.z - nose_tip.z)))
    yaw = np.degrees(np.arctan2(right_eye_outer.x - left_eye_outer.x, right_eye_outer.z - left_eye_outer.z))
    roll = np.degrees(np.arctan2(forehead.y - chin.y, forehead.x - chin.x))

    return pitch, yaw, roll


def compute_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR) to detect drowsiness"""
    vertical1 = np.linalg.norm(
        np.array([landmarks[159].x, landmarks[159].y]) - 
        np.array([landmarks[145].x, landmarks[145].y])
    )
    vertical2 = np.linalg.norm(
        np.array([landmarks[158].x, landmarks[158].y]) - 
        np.array([landmarks[144].x, landmarks[144].y])
    )
    horizontal = np.linalg.norm(
        np.array([landmarks[33].x, landmarks[33].y]) - 
        np.array([landmarks[133].x, landmarks[133].y])
    )
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_center(landmarks, indices):
    """Calculate the center of a set of landmarks"""
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    return np.mean(points, axis=0)


def update_frame():
    global baseline_set, baseline_pitch, baseline_yaw, baseline_roll
    global gaze_start_time, gaze_alert_triggered, head_alert_start_time, head_alert_triggered

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    abnormal_alert = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if not baseline_set:
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark)
                elapsed_time = time.time() - start_time
                baseline_data.append((pitch, yaw, roll))

                if elapsed_time >= threshold_time:
                    baseline_pitch, baseline_yaw, baseline_roll = np.mean(baseline_data, axis=0)
                    baseline_set = True
            else:
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark)

                if abs(pitch - baseline_pitch) > PITCH_THRESHOLD or pitch>74:
                    if head_alert_start_time is None:
                        head_alert_start_time = time.time()
                    elif time.time() - head_alert_start_time >= head_abnormal_duration:
                        abnormal_alert = "ABNORMAL PITCH!"
                elif abs(yaw - baseline_yaw) > YAW_THRESHOLD:
                    if head_alert_start_time is None:
                        head_alert_start_time = time.time()
                    elif time.time() - head_alert_start_time >= head_abnormal_duration:
                        abnormal_alert = "ABNORMAL YAW!"
                elif abs(roll - baseline_roll) > ROLL_THRESHOLD:
                    if head_alert_start_time is None:
                        head_alert_start_time = time.time()
                    elif time.time() - head_alert_start_time >= head_abnormal_duration:
                        abnormal_alert = "ABNORMAL ROLL!"
                else:
                    head_alert_start_time = None

                pitch_label.config(text=f"Pitch: {pitch:.2f} deg")
                yaw_label.config(text=f"Yaw: {yaw:.2f} deg")
                roll_label.config(text=f"Roll: {roll:.2f} deg")

            # Gaze Detection
            left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
            left_iris_indices = [468, 469, 470, 471]

            left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
            left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

            left_eye_width = np.linalg.norm(
                np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
            )
            left_iris_position_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width
            left_ear = compute_ear(face_landmarks.landmark, left_eye_indices)

            gaze = "Center"
            if left_iris_position_x < -0.1:
                gaze = "Right"
            elif left_iris_position_x > 0.1:
                gaze = "Left"
            elif left_ear < EAR_THRESHOLD:
                gaze = "Down"

            if gaze in ["Left", "Right", "Down"]:
                if gaze_start_time is None:
                    gaze_start_time = time.time()
                elif time.time() - gaze_start_time >= gaze_abnormal_duration:
                    gaze_alert_triggered = True
            else:
                gaze_start_time = None
                gaze_alert_triggered = False

            if gaze_alert_triggered and abnormal_alert is None:
                abnormal_alert = "ABNORMAL GAZE!"

            gaze_label.config(text=f"Gaze: {gaze}")

    alert_label.config(text=abnormal_alert if abnormal_alert else "", fg="red" if abnormal_alert else "black")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
