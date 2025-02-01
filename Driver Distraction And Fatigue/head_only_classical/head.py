import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

PITCH_THRESHOLD = 15
YAW_THRESHOLD = 15
ROLL_THRESHOLD = 15

def get_head_pose(landmarks, frame_width, frame_height):
    image_points = np.array([
        (landmarks[1].x * frame_width, landmarks[1].y * frame_height),   # Nose tip
        (landmarks[152].x * frame_width, landmarks[152].y * frame_height), # Chin
        (landmarks[33].x * frame_width, landmarks[33].y * frame_height),  # Left eye corner
        (landmarks[263].x * frame_width, landmarks[263].y * frame_height), # Right eye corner
        (landmarks[61].x * frame_width, landmarks[61].y * frame_height),  # Left mouth corner
        (landmarks[291].x * frame_width, landmarks[291].y * frame_height)  # Right mouth corner
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),    # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-34.29, 20.94, -21.64),  # Left eye corner
        (34.29, 20.94, -21.64),  # Right eye corner
        (-21.1, -21.72, -21.1),  # Left mouth corner
        (21.1, -21.72, -21.1)   # Right mouth corner
    ], dtype=np.float64)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Check and extract 3x3 rotation matrix
        if rotation_matrix.shape == (3, 3):
            pass
        elif rotation_matrix.shape == (4, 3):
            rotation_matrix = rotation_matrix[:3, :3]  # Extract the 3x3 part if it's 4x3

        # Decompose the 3x3 rotation matrix
        euler_angles, _, _, _, _, _ = cv2.decomposeProjectionMatrix(rotation_matrix)
        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll
    return None, None, None

def detect_abnormal_movement(pitch, yaw, roll, pitch_baseline, yaw_baseline, roll_baseline):
    abnormal = []
    if abs(pitch - pitch_baseline) > PITCH_THRESHOLD:
        abnormal.append("Pitch")
    if abs(yaw - yaw_baseline) > YAW_THRESHOLD:
        abnormal.append("Yaw")
    if abs(roll - roll_baseline) > ROLL_THRESHOLD:
        abnormal.append("Roll")
    return abnormal

cap = cv2.VideoCapture(0)
pitch_baseline, yaw_baseline, roll_baseline = 0, 0, 0
baseline_frames = 50
frame_count = 0
start_time = None
abnormal_detected = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pitch, yaw, roll = get_head_pose(face_landmarks.landmark, frame_width, frame_height)
            if pitch is not None and yaw is not None and roll is not None:
                if frame_count < baseline_frames:
                    pitch_baseline += pitch
                    yaw_baseline += yaw
                    roll_baseline += roll
                    frame_count += 1
                    if frame_count == baseline_frames:
                        pitch_baseline /= baseline_frames
                        yaw_baseline /= baseline_frames
                        roll_baseline /= baseline_frames
                else:
                    abnormal_movements = detect_abnormal_movement(pitch, yaw, roll, pitch_baseline, yaw_baseline, roll_baseline)
                    if abnormal_movements:
                        if start_time is None:
                            start_time = time.time()
                            abnormal_detected = abnormal_movements
                        elif time.time() - start_time >= 5:
                            cv2.putText(frame, f"Abnormal: {', '.join(abnormal_detected)}", (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        start_time = None
                        abnormal_detected = None

    cv2.imshow('Head Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
