import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def calculate_pitch(nose, chin):
    """Compute the pitch angle using nose and chin landmarks."""
    # Vector from nose to chin
    vector = np.array([chin[0] - nose[0], chin[1] - nose[1], chin[2] - nose[2]])
    
    # Reference vector along Y-axis (assuming the camera's coordinate system)
    reference_vector = np.array([0, -1, 0])  # Pointing downward

    # Normalize vectors
    vector = vector / np.linalg.norm(vector)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    # Compute angle using dot product
    dot_product = np.dot(vector, reference_vector)
    angle = np.arccos(dot_product)  # In radians

    # Convert to degrees
    pitch_angle = np.degrees(angle) - 90  # Adjusting to align with natural head tilt
    return pitch_angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract 3D coordinates (normalized)
            nose = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]

            # Convert to pixel coordinates
            nose_3d = np.array([nose.x * w, nose.y * h, nose.z * w])
            chin_3d = np.array([chin.x * w, chin.y * h, chin.z * w])

            # Compute pitch
            pitch = calculate_pitch(nose_3d, chin_3d)

            # Display result
            cv2.putText(frame, f'Pitch: {pitch:.2f} deg', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Head Pitch Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
