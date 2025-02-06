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
)  # Configure Face Mesh to detect one face with high confidence
mp_drawing = mp.solutions.drawing_utils  # Mediapipe utility to draw landmarks

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
threshold_time = 15


baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0
baseline_data = []
baseline_set = False

results =0 
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

def calculate_pitch(nose, chin):
    """Compute the pitch angle using nose and chin landmarks."""
    # Vector from nose to chin (3D)
    #positive value means the chin is lower than the nose (head tilted downward "forward")
    #negative value means the chin is above the nose (head tilted upward "backword")
    vector = np.array([chin[0] - nose[0], chin[1] - nose[1], chin[2] - nose[2]])

    # Compute pitch using atan2 (Preserves sign for up/down movement)
    #projection of the vector in the X-Z plane "np.linalg.norm([vector[0], vector[2]])"
    pitch_angle = np.degrees(np.arctan2(vector[1], np.linalg.norm([vector[0], vector[2]])))

    # Adjust for backward movement
    if chin[2] > nose[2]:  # If chin is deeper into the screen than the nose
        pitch_angle *= -1   # Invert pitch to reflect backward movement correctly

    return pitch_angle

# Head Movement Functions
def calculate_angles(face_landmarks, frame_width, frame_height):  # Add face_landmarks as parameter
    # Select key points for calculating angles
    nose_tip = face_landmarks.landmark[1]  # Nose tip landmark
    chin = face_landmarks.landmark[152]  # Chin landmark
    left_eye_outer = face_landmarks.landmark[33]  # Outer corner of the left eye
    right_eye_outer = face_landmarks.landmark[263]  # Outer corner of the right eye
    forehead = face_landmarks.landmark[10]  # Forehead landmark

    # Convert normalized landmarks to pixel coordinates
    def normalized_to_pixel(normalized, width, height):
        return int(normalized.x * width), int(normalized.y * height)

    # Convert normalized coordinates to pixel coordinates for selected landmarks
    nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
    chin = normalized_to_pixel(chin, frame_width, frame_height)
    left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
    right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
    forehead = normalized_to_pixel(forehead, frame_width, frame_height)

    # Extract 3D coordinates (normalized)
    nose_for_pitch = face_landmarks.landmark[1]
    chin_for_pitch = face_landmarks.landmark[152]
    # Convert to pixel coordinates
    nose_3d = np.array([nose_for_pitch.x * frame_width, nose_for_pitch.y * frame_height, nose_for_pitch.z * frame_width])
    chin_3d = np.array([chin_for_pitch.x * frame_width, chin_for_pitch.y * frame_height, chin_for_pitch.z * frame_width])

    # Compute pitch using 3D coordinates
    pitch = calculate_pitch(nose_3d, chin_3d)

    # Calculate yaw (horizontal head angle) based on the eye positions
    delta_x_eye = right_eye_outer[0] - left_eye_outer[0]
    delta_y_eye = right_eye_outer[1] - left_eye_outer[1]
    yaw = np.arctan2(delta_y_eye, delta_x_eye) * (180 / np.pi)

    # Calculate roll (head tilt angle) based on the forehead and chin positions
    delta_x_forehead = forehead[0] - chin[0]
    delta_y_forehead = forehead[1] - chin[1]
    roll = np.arctan2(delta_y_forehead, delta_x_forehead) * (180 / np.pi)

    return pitch, yaw, roll  # Return calculated angles


def check_abnormal_angles(pitch, yaw, roll):
    if abs(pitch - baseline_pitch) > PITCH_THRESHOLD or pitch > 73:
        return "Abnormal Pitch"
    if abs(yaw - baseline_yaw) > YAW_THRESHOLD:
        return "Abnormal Yaw"
    if abs(roll - baseline_roll) > ROLL_THRESHOLD:
        return "Abnormal Roll"
    return None

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
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark, w, h)
                elapsed_time = time.time() - start_time
                baseline_data.append((pitch, yaw, roll))
                if elapsed_time >= threshold_time:
                    baseline_pitch, baseline_yaw, baseline_roll = np.mean(baseline_data, axis=0)
                    baseline_set = True
            else:
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark, w, h)
                abnormal_alert = check_abnormal_angles(pitch, yaw, roll)
                
                if abnormal_alert:
                    if head_alert_start_time is None:
                        head_alert_start_time = time.time()
                    elif time.time() - head_alert_start_time > head_abnormal_duration:
                        head_alert_triggered = True
                else:
                    head_alert_start_time = None
                    head_alert_triggered = False

                if head_alert_triggered:
                    alert_label.config(text=abnormal_alert, fg="red")
                else:
                    alert_label.config(text="", fg="black")

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
