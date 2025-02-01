import cv2  # OpenCV for video processing
import mediapipe as mp  # Mediapipe for face mesh and landmark detection
import numpy as np  # NumPy for mathematical calculations
import time  # Time module for timers

# Initialize Mediapipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # Configure Face Mesh to detect one face with high confidence
mp_drawing = mp.solutions.drawing_utils  # Mediapipe utility to draw landmarks

# Initialize timers and state variables
start_time = time.time()  # Start timer to establish baseline for head movements
threshold_time = 10  # Duration in seconds to establish a baseline
baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0  # Initialize baseline angles for head movement
baseline_data = []  # Store head movement angles for baseline calculation
baseline_set = False  # Flag to indicate whether baseline is established

# Global variables for gaze and head movement detection
gaze_start_time = None  # Start time for abnormal gaze detection
gaze_alert_triggered = False  # Flag to indicate abnormal gaze
gaze_abnormal_duration = 5  # Duration (in seconds) to trigger abnormal gaze alert

head_alert_start_time = None  # Start time for abnormal head movement detection
head_alert_triggered = False  # Flag to indicate abnormal head movement
head_abnormal_duration = 5  # Duration (in seconds) to trigger abnormal head movement alert

# Thresholds for detecting abnormal head movements
PITCH_THRESHOLD = 4  # Angle in degrees for abnormal pitch
YAW_THRESHOLD = 15  # Angle in degrees for abnormal yaw
ROLL_THRESHOLD = 15  # Angle in degrees for abnormal roll

# Head Movement Functions
def calculate_angles(landmarks, frame_width, frame_height):
    # Select key points for calculating angles
    nose_tip = landmarks[1]  # Nose tip landmark
    chin = landmarks[152]  # Chin landmark
    left_eye_outer = landmarks[33]  # Outer corner of the left eye
    right_eye_outer = landmarks[263]  # Outer corner of the right eye
    forehead = landmarks[10]  # Forehead landmark

    # Convert normalized landmarks to pixel coordinates
    def normalized_to_pixel(normalized, width, height):
        return int(normalized.x * width), int(normalized.y * height)

    # Convert normalized coordinates to pixel coordinates for selected landmarks
    nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
    chin = normalized_to_pixel(chin, frame_width, frame_height)
    left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
    right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
    forehead = normalized_to_pixel(forehead, frame_width, frame_height)

    # Calculate pitch (vertical head angle) based on the difference between chin and nose tip
    delta_y = chin[1] - nose_tip[1]
    delta_x = chin[0] - nose_tip[0]
    pitch = np.arctan2(delta_y, delta_x) * (180 / np.pi)  # Convert from radians to degrees

    # Calculate yaw (horizontal head angle) based on the eye positions
    delta_x_eye = right_eye_outer[0] - left_eye_outer[0]
    delta_y_eye = right_eye_outer[1] - left_eye_outer[1]
    yaw = np.arctan2(delta_y_eye, delta_x_eye) * (180 / np.pi)

    # Calculate roll (head tilt angle) based on the forehead and chin positions
    delta_x_forehead = forehead[0] - chin[0]
    delta_y_forehead = forehead[1] - chin[1]
    roll = np.arctan2(delta_y_forehead, delta_x_forehead) * (180 / np.pi)

    return pitch, yaw, roll  # Return calculated angles

# Function to check if head movement angles exceed thresholds
def check_abnormal_angles(pitch, yaw, roll, movement_type):
    alerts = []  # List to store abnormal movement alerts
    if movement_type == 'pitch':  # Check for abnormal pitch
        alerts.append("Abnormal Pitch")
    if movement_type == 'yaw' :  # Check for abnormal yaw
        alerts.append("Abnormal Yaw")
    elif movement_type == 'roll':  # Check for abnormal roll
        alerts.append("Abnormal Roll")
    
    return alerts  # Return list of alerts

# Open webcam
cap = cv2.VideoCapture(0)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:  # Break loop if the frame cannot be read
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Mediapipe processing
    results = face_mesh.process(rgb_frame)  # Process the frame with Mediapipe Face Mesh

    if results.multi_face_landmarks:  # Check if any face is detected
        for face_landmarks in results.multi_face_landmarks:  # Process each detected face

            # If baseline is not set, collect data to calculate baseline angles
            if not baseline_set:
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark, frame.shape[1], frame.shape[0])
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                baseline_data.append((pitch, yaw, roll))  # Append current angles to baseline data
                if elapsed_time >= threshold_time:  # Check if threshold time is reached
                    # Calculate baseline as the average of collected angles
                    baseline_pitch, baseline_yaw, baseline_roll = np.mean(baseline_data, axis=0)
                    baseline_set = True  # Set baseline flag to True
                # Display message during baseline setting
                cv2.putText(frame, "Setting baseline...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Check for abnormal head movements
                # Calculate head movement angles
                pitch, yaw, roll = calculate_angles(face_landmarks.landmark, frame.shape[1], frame.shape[0])
                head_alerts = []

                # Only check the angle that exceeds the threshold
                if abs(pitch - baseline_pitch) > PITCH_THRESHOLD:
                    head_alerts = check_abnormal_angles(pitch, yaw, roll, 'pitch')  # Check for abnormal pitch
                if abs(yaw - baseline_yaw) > YAW_THRESHOLD:
                    head_alerts = check_abnormal_angles(pitch, yaw, roll, 'yaw')  # Check for abnormal yaw
                if abs(roll - baseline_roll) > ROLL_THRESHOLD:
                    head_alerts = check_abnormal_angles(pitch, yaw, roll, 'roll')  # Check for abnormal roll

                if head_alerts:  # If any abnormal angles are detected
                    if head_alert_start_time is None:  # Start timer for abnormal movement
                        head_alert_start_time = time.time()
                    elif time.time() - head_alert_start_time > head_abnormal_duration and not head_alert_triggered:
                        head_alert_triggered = True  # Trigger abnormal head movement alert
                else:
                    head_alert_start_time = None  # Reset timer if no abnormal movement
                    head_alert_triggered = False

                # Display abnormal head movement alerts if triggered
                if head_alert_triggered:
                    for i, alert in enumerate(head_alerts):  # Display each alert
                        cv2.putText(frame, alert, (10, 120 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Display head movement angles on the top-left corner
                cv2.putText(frame, f"Pitch: {pitch:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Eye landmarks for gaze detection
            left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]  # Left eye landmarks
            left_iris_indices = [468, 469, 470, 471]  # Left iris landmarks

            # Helper function to calculate the center of a set of landmarks
            def get_center(landmarks, indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])  # Extract landmark points
                return np.mean(points, axis=0)  # Return the mean of the points

            # Calculate centers for the left eye and iris
            left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
            left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

            # Calculate the relative horizontal position of the iris
            left_eye_width = np.linalg.norm(
                np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
            )
            left_iris_position = (left_iris_center[0] - left_eye_center[0]) / left_eye_width  # Normalize position

            # Determine gaze direction
            gaze = "Center"
            if left_iris_position < -0.1:  # If iris position is on the left side
                gaze = "Right"
            elif left_iris_position > 0.1:  # If iris position is on the right side
                gaze = "Left"

            # Check for abnormal gaze direction
            if gaze in ["Left", "Right"]:  # If gaze is not centered
                if gaze_start_time is None:  # Start timer for abnormal gaze
                    gaze_start_time = time.time()
                elif time.time() - gaze_start_time > gaze_abnormal_duration and not gaze_alert_triggered:
                    gaze_alert_triggered = True  # Trigger abnormal gaze alert
            else:
                gaze_start_time = None  # Reset gaze timer
                gaze_alert_triggered = False

            # Display abnormal gaze alert if triggered
            if gaze_alert_triggered:
                cv2.putText(frame, "ABNORMAL GAZE", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display gaze direction on the top-right corner
            cv2.putText(frame, f"Gaze: {gaze}", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the video frame with all annotations
    cv2.imshow("Head Movement and Gaze Detection", frame)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
