import cv2  # OpenCV for video processing
import mediapipe as mp  # Mediapipe for face mesh and landmark detection
import numpy as np  # NumPy for mathematical calculations
import time  # Time module for timers
import platform
import threading
import tkinter as tk
from PIL import Image, ImageTk



####################################-----GUI-----#############################################
pitch_gui= 0.0
yaw_gui=0.0
roll_gui= 0.0
gaze_gui= "Center"
gaze_status_gui= 0
head_status_gui= 0
flag_gui=0
distraction_flag_head=0
distraction_flag_gaze=0
temp =0 
temp_g=0

# Create the GUI window
root = tk.Tk()
root.title("Webcam GUI")
root.geometry("980x600")  # Adjust window size as needed

# Create a frame for the title
title_label = tk.Label(
    root, text="GAZE AND HEAD DETECTION", 
    font=("Courier New", 24, "bold"), fg="blue"
)
title_label.pack(pady=10)

# Create a frame for layout
frame_container = tk.Frame(root)
frame_container.pack(fill="both", expand=True)

# Create a label for the video feed (on the left side)
video_label = tk.Label(frame_container)
video_label.pack(side="left", padx=10, pady=10)

# Create a frame for text output (on the right side)
text_frame = tk.Frame(frame_container)
text_frame.pack(side="left", padx=30, pady=30)

# GAZE Section
gaze_label = tk.Label(text_frame, text="GAZE DETECTION", font=("Arial", 18, "bold"))
gaze_label.pack(anchor="w")

gaze_center_label = tk.Label(text_frame, text="Center: ", font=("Arial", 14))
gaze_center_label.pack(anchor="w")

gaze_status_label = tk.Label(text_frame, text="Status: ", font=("Arial", 14))
gaze_status_label.pack(anchor="w")

# HEAD Section
head_label = tk.Label(text_frame, text="HEAD DETETCION", font=("Arial", 18, "bold"))
head_label.pack(anchor="w", pady=(10, 0))

pitch_label = tk.Label(text_frame, text="Pitch: ", font=("Arial", 14))
pitch_label.pack(anchor="w")

yaw_label = tk.Label(text_frame, text="Yaw: ", font=("Arial", 14))
yaw_label.pack(anchor="w")

roll_label = tk.Label(text_frame, text="Roll: ", font=("Arial", 14))
roll_label.pack(anchor="w")

head_status_label = tk.Label(text_frame, text="Status: ", font=("Arial", 14))
head_status_label.pack(anchor="w")

# Add a separator (optional)
separator = tk.Label(text_frame, text="--------------------------------------------------", font=("Arial", 14))
separator.pack(anchor="w", pady=10)  # Adds a line break for separation


distraction_label = tk.Label(text_frame, text="Distraction counts within 3 min : ", font=("Arial", 14))
distraction_label.pack(anchor="w")

# Add a separator (optional)
separator = tk.Label(text_frame, text="--------------------------------------------------", font=("Arial", 14))
separator.pack(anchor="w", pady=10)  # Adds a line break for separation

distraction_flag_label = tk.Label(text_frame, text="", font=("Arial", 18, "bold"))
distraction_flag_label.pack(anchor="w")


distraction_highriskflag_label = tk.Label(text_frame, text="", font=("Arial", 18, "bold"))
distraction_highriskflag_label.pack(anchor="w")


##############################################################################################

# Initialize Mediapipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # Configure Face Mesh to detect one face with high confidence
mp_drawing = mp.solutions.drawing_utils  # Mediapipe utility to draw landmarks

# Initialize timers and state variables
start_time = time.time()  # Start timer to establish baseline for head movements
threshold_time = 20       # Duration in seconds to establish a baseline
baseline_pitch, baseline_yaw, baseline_roll = 0, 0, 0  # Initialize baseline angles for head movement
baseline_data = []     # Store head movement angles for baseline calculation
baseline_set = False   # Flag to indicate whether baseline is established


# Global variables for gaze and head movement detection
gaze_start_time = None        # Start time for abnormal gaze detection
gaze_alert_triggered = False  # Flag to indicate abnormal gaze
gaze_abnormal_duration = 5    # Duration (in seconds) to trigger abnormal gaze alert

head_alert_start_time = None  # Start time for abnormal head movement detection
head_alert_triggered = False  # Flag to indicate abnormal head movement
head_abnormal_duration = 5    # Duration (in seconds) to trigger abnormal head movement alert

# Thresholds for detecting abnormal head movements
PITCH_THRESHOLD = 10         # Angle in degrees for abnormal pitch
YAW_THRESHOLD = 10           # Angle in degrees for abnormal yaw
ROLL_THRESHOLD = 10          # Angle in degrees for abnormal roll
EAR_THRESHOLD = 0.35         # EAR threshold which below it considered looking down
NO_BLINK_GAZE_DURATION = 20  # Time (seconds) for center gaze without blinking to be considered abnormal


# Global buzzer control
buzzer_running = False
no_blink_start_time = None  # Timer for no blink detection



distraction_counter = 0
#time_limit_counter = 100  #  minutes in seconds
start_time_counter = time.time()  # Initialize start time
DISTRACTION_THRESHOLD=4




#################################-----Used Functions----##########################################
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


def buzzer_alert():
    global buzzer_running
    if buzzer_running:
        return  # Prevent multiple buzzer threads
    buzzer_running = True
    
    def play_buzzer():
        while buzzer_running:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
            else:
                import os
                os.system('play -nq -t alsa synth 0.5 sine 500')  # Works on Linux with sox installed
    
    buzzer_thread = threading.Thread(target=play_buzzer, daemon=True)
    buzzer_thread.start()

def stop_buzzer():
    global buzzer_running
    buzzer_running = False


def compute_ear(landmarks, eye_indices):
    # Vertical distances
    #Lower and upper eyelid of the left eye
    vertical1 = np.linalg.norm(
        np.array([landmarks[159].x, landmarks[159].y]) - 
        np.array([landmarks[145].x, landmarks[145].y])
    )
    #Lower and upper eyelid of the right eye
    vertical2 = np.linalg.norm(
        np.array([landmarks[158].x, landmarks[158].y]) - 
        np.array([landmarks[144].x, landmarks[144].y])
    )
    # Horizontal distance
    #Outer and inner corners of the left eye
    horizontal = np.linalg.norm(
        np.array([landmarks[33].x, landmarks[33].y]) - 
        np.array([landmarks[133].x, landmarks[133].y])
    )
    # Compute EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


# Function to detect abnormal center gaze "without blinking"
def process_blink_and_gaze(gaze, left_ear , left_iris_position_y):
    global no_blink_start_time, gaze_alert_triggered
    
    if gaze == "Center":
        if no_blink_start_time is None:
            no_blink_start_time = time.time()
            
        else:
            elapsed_time = time.time() - no_blink_start_time
            if elapsed_time >= NO_BLINK_GAZE_DURATION:
                if not gaze_alert_triggered:
                    gaze_alert_triggered = True  # Consider prolonged center gaze without blinking as abnormal
                    gaze = "Center Gazed"        # Set gaze to "Center Gazed" to trigger alert
                    
    else:
        no_blink_start_time = None    # Reset timer if gaze changes
    
    if left_iris_position_y < -0.3 and left_ear < EAR_THRESHOLD:
        no_blink_start_time = None    # Reset blink timer if blink is detected
        gaze_alert_triggered = False  # Reset abnormal gaze trigger if blinking occurs
        gaze = "Down"
    return gaze

def calculate_angles(landmarks, frame_width, frame_height):
    
    # Select key points for calculating angles
    nose_tip = landmarks[1]           # Nose tip landmark
    chin = landmarks[152]             # Chin landmark
    left_eye_outer = landmarks[33]    # Outer corner of the left eye
    right_eye_outer = landmarks[263]  # Outer corner of the right eye
    forehead = landmarks[10]          # Forehead landmark

    # Convert normalized landmarks to pixel coordinates
    def normalized_to_pixel(normalized, width, height):
        return int(normalized.x * width), int(normalized.y * height)

    # Convert normalized coordinates to pixel coordinates for selected landmarks
    nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
    chin = normalized_to_pixel(chin, frame_width, frame_height)
    left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
    right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
    forehead = normalized_to_pixel(forehead, frame_width, frame_height)

    # Extract 3D coordinates (normalized) correctly using the passed `landmarks` variable
    nose_for_pitch = landmarks[1]
    chin_for_pitch = landmarks[152]

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


# Function to check if head movement angles exceed thresholds
def check_abnormal_angles(pitch, yaw, roll, movement_type):
    alerts = []                     
    if movement_type == 'pitch':    
        alerts.append("Abnormal Pitch")
    if movement_type == 'yaw' :     
        alerts.append("Abnormal Yaw")
    elif movement_type == 'roll':  
        alerts.append("Abnormal Roll")
    return alerts                   

def action_after_buzzer():
    global temp , temp_g
    temp=0
    temp_g=0

def reset_distraction_flag():
    global distraction_flag_head, distraction_flag_gaze, distraction_counter, start_time_counter

    # Reset distraction flags
    distraction_flag_head = 0  
    distraction_flag_gaze = 0  

    # Reset distraction counter after buzzer stops
    distraction_counter = 0  
    start_time_counter = time.time()  

    # Run action_after_buzzer() after 1 second
    root.after(1000, action_after_buzzer)


###################################----start the process----##################################


# Capture video from webcam
cap = cv2.VideoCapture(0)

# Process video frames
def process_video():
    """
    Processes video frames in real-time to monitor head movement and gaze direction.
    Updates global variables and triggers warnings based on detected distractions.
    """
    # Declare global variables
    global baseline_set, head_alert_triggered, gaze_abnormal_duration, gaze_alert_triggered, gaze_center_label, gaze_start_time
    global head_alert_start_time
    global pitch_gui, yaw_gui, roll_gui, head_status_gui
    global gaze_gui, gaze_status_gui
    global flag_gui
    global no_blink_start_time, gaze_alert_triggered
    global distraction_flag_gaze, distraction_flag_head
    global distraction_counter
    global temp, temp_g

    # Initialize distraction time counter
    start_time_counter = time.time()  # Start tracking distractions
    time_limit_counter = 180  # 3-minute threshold

    while cap.isOpened():
        # Calculate elapsed time for distraction tracking
        elapsed_time_counter = time.time() - start_time_counter

        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:  # Exit if no frame is received
            break

        # Convert frame to RGB format for Mediapipe processing
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)  # Process frame using Mediapipe Face Mesh

        # -------------------------------------------- Head Movement Detection --------------------------------------------
        if results.multi_face_landmarks:  # If a face is detected
            for face_landmarks in results.multi_face_landmarks:

                # ✅ Step 1: Collect baseline angles if not set
                if not baseline_set:
                    pitch, yaw, roll = calculate_angles(face_landmarks.landmark, w, h)
                    elapsed_time = time.time() - start_time  # Track baseline time
                    baseline_data.append((pitch, yaw, roll))  # Store collected angles

                    # ✅ Once baseline is collected for the threshold duration
                    if elapsed_time >= threshold_time:
                        # Compute baseline averages
                        baseline_pitch, baseline_yaw, baseline_roll = np.mean(baseline_data, axis=0)
                        baseline_set = True  # Mark baseline as set

                        # ✅ Start GUI updates
                        if flag_gui == 0:
                            root.after(100, update_text) 

                else:
                    # ✅ Step 2: Calculate head angles
                    flag_gui = 1  # Enable GUI updates
                    pitch, yaw, roll = calculate_angles(face_landmarks.landmark, w, h)
                    pitch_gui, yaw_gui, roll_gui = pitch, yaw, roll  # Update GUI values

                    # Detect abnormal movements
                    head_alerts = []
                    if abs(pitch - baseline_pitch) > PITCH_THRESHOLD or pitch > 73:
                        head_alerts = check_abnormal_angles(pitch, yaw, roll, 'pitch')
                    if abs(yaw - baseline_yaw) > YAW_THRESHOLD:
                        head_alerts = check_abnormal_angles(pitch, yaw, roll, 'yaw')
                    if abs(roll - baseline_roll) > ROLL_THRESHOLD:
                        head_alerts = check_abnormal_angles(pitch, yaw, roll, 'roll')

                    # ✅ Step 3: Trigger alerts based on detected movements
                    if head_alerts:
                        if head_alert_start_time is None:  # Start timer
                            head_alert_start_time = time.time()
                        elif time.time() - head_alert_start_time > head_abnormal_duration and not head_alert_triggered:
                            head_alert_triggered = True
                            distraction_counter += 1  # Increase distraction counter
                    else:
                        head_alert_start_time = None  # Reset timer
                        head_alert_triggered = False

                    # ✅ Step 4: Update GUI warnings
                    if head_alert_triggered:
                        if "Abnormal Pitch" in head_alerts:
                            head_status_gui = "ABNORMAL PITCH"
                            distraction_flag_head = 1
                        elif "Abnormal Yaw" in head_alerts:
                            head_status_gui = "ABNORMAL YAW"
                            distraction_flag_head = 1
                        elif "Abnormal Roll" in head_alerts:
                            head_status_gui = "ABNORMAL ROLL"
                            distraction_flag_head = 1
                    else:
                        head_status_gui = "NORMAL"
                        distraction_flag_head = 0

        # -------------------------------------------- Gaze Detection --------------------------------------------
                # Define facial landmarks for eye and iris detection
                left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
                left_iris_indices = [468, 469, 470, 471]

                # Helper function to calculate the center of landmarks
                def get_center(landmarks, indices):
                    points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                    return np.mean(points, axis=0)  # Compute mean center

                # Compute eye and iris centers
                left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
                left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

                # Compute iris horizontal position
                left_eye_width = np.linalg.norm(
                    np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                    np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
                )
                left_iris_position_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width

                # Compute iris vertical position
                left_eye_height = np.linalg.norm(
                    np.array([face_landmarks.landmark[159].x, face_landmarks.landmark[159].y]) - 
                    np.array([face_landmarks.landmark[145].x, face_landmarks.landmark[145].y])
                )
                left_iris_position_y = (left_iris_center[1] - left_eye_center[1]) / left_eye_height

                # ✅ Step 1: Detect Gaze Direction
                if left_iris_position_x < -0.1:
                    gaze = "Right"
                elif left_iris_position_x > 0.1:
                    gaze = "Left"
                else:
                    gaze = process_blink_and_gaze("Center", compute_ear(face_landmarks.landmark, left_eye_indices), left_iris_position_y)
                
                gaze_gui = gaze  # Update GUI

                # ✅ Step 2: Detect Abnormal Gaze
                if gaze in ["Left", "Right", "Down", "Center Gazed"]:
                    if gaze_start_time is None:
                        gaze_start_time = time.time()
                    elif time.time() - gaze_start_time > gaze_abnormal_duration and not gaze_alert_triggered:
                        gaze_alert_triggered = True
                        distraction_counter += 1
                else:
                    gaze_start_time = None
                    gaze_alert_triggered = False

                # ✅ Step 3: Update Gaze Warnings
                if gaze_alert_triggered:
                    gaze_status_gui = "ABNORMAL GAZE"
                    distraction_flag_gaze = 1
                else:
                    gaze_status_gui = "NORMAL"
                    distraction_flag_gaze = 0

        # -------------------------------------------- Distraction Handling --------------------------------------------


                # ✅ If distraction threshold is reached, trigger HIGH RISK alert
                if distraction_counter >= DISTRACTION_THRESHOLD and elapsed_time_counter < time_limit_counter:
                    temp = 1
                    temp_g = 1
                    distraction_flag_head = 2
                    distraction_flag_gaze = 2

                    # Activate buzzer alert
                    buzzer_alert()
                    
                    # Stop buzzer after 5 sec and reset distractions
                    root.after(5000, stop_buzzer)
                    root.after(9000, reset_distraction_flag)
                elif elapsed_time_counter >= time_limit_counter:
                    # ✅ Reset counter every 3 minutes
                    print("⏳ 3 minutes passed. Resetting counter.")
                    distraction_counter = 0
                    start_time_counter = time.time()

        # ✅ Display the processed frame
        cv2.imshow("Head Movement and Gaze Detection", frame)




def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (500, 400))  
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_frame)  # Keep calling itself for continuous update

def update_text():
    """
    Updates the GUI labels based on the distraction status (head and gaze).
    Runs continuously every 500ms to refresh the displayed information.
    """
    global flag_gui, distraction_flag_gaze, distraction_flag_head, temp, temp_g

    # Define colors
    dark_orange = "#FF8C00"  # Warning color
    grey = "#808080"  # Default color for baseline setting
    warning_text = ""  # Initialize empty warning message
    warning_color = "black"  # Default text color

    ### 📌 Baseline Setup (Before Detection Starts) ###
    if flag_gui == 0:
        # Display "Setting Baseline..." for all metrics
        pitch_gui_t = yaw_gui_t = roll_gui_t = gaze_gui_t = "setting baseline.."
        gaze_status_gui_t = head_status_gui_t = "setting baseline.."

        # Update GUI labels
        gaze_center_label.config(text=f"Position: {gaze_gui_t}", fg=grey)
        gaze_status_label.config(text=f"Gaze Status: {gaze_status_gui_t}", fg=grey)
        pitch_label.config(text=f"Pitch: {pitch_gui_t} deg", fg=grey)
        yaw_label.config(text=f"Yaw: {yaw_gui_t} deg", fg=grey)
        roll_label.config(text=f"Roll: {roll_gui_t} deg", fg=grey)
        head_status_label.config(text=f"Head Status: {head_status_gui_t}", fg=grey)

    ### 📌 Detection & Warnings (Active Monitoring) ###
    elif flag_gui == 1:
        # Update gaze position
        gaze_center_label.config(text=f"Position: {gaze_gui}", fg="black")

        # Display gaze status (Normal / Abnormal)
        if gaze_status_gui == "NORMAL":
            gaze_status_label.config(text=f"Gaze Status: {gaze_status_gui} ✅", fg="green")  # Normal gaze
        else:
            gaze_status_label.config(text=f"Gaze Status: {gaze_status_gui} 🚨", fg="red")  # Abnormal gaze

        # Update head movement values
        pitch_label.config(text=f"Pitch: {pitch_gui:.2f} deg", fg="black")
        yaw_label.config(text=f"Yaw: {yaw_gui:.2f} deg", fg="black")
        roll_label.config(text=f"Roll: {roll_gui:.2f} deg", fg="black")

        # Display head status (Normal / Abnormal)
        if head_status_gui == "NORMAL":
            head_status_label.config(text=f"Head Status: {head_status_gui} ✅", fg="green")
        else:
            head_status_label.config(text=f"Head Status: {head_status_gui} 🚨", fg="red")

        # Update distraction counter
        distraction_label.config(text=f"Distraction Count within 3 min: {distraction_counter}", fg="black")

        ### 📌 Checking Distraction Warnings ###

        # 🛑 Gaze Warnings
        if distraction_flag_gaze == 2 and temp_g == 1:  # High risk due to gaze
            warning_text += "🚨 HIGH RISK 🚨"
            warning_color = "red"
        elif distraction_flag_gaze == 1 and temp_g == 0:  # Moderate distraction due to gaze
            warning_text += "⚠️ WARNING: Driver Distracted!"
            warning_color = dark_orange

        # 🛑 Head Movement Warnings
        if distraction_flag_head == 2 and temp == 1:  # High risk due to head movement
            warning_text += "🚨 HIGH RISK 🚨"
            warning_color = "red"
        elif distraction_flag_head == 1 and temp == 0:  # Moderate distraction due to head movement
            warning_text += "⚠️ WARNING: Driver Distracted!"
            warning_color = dark_orange

        ### 📌 Updating the Warning Label ###
        if warning_text == "":
            distraction_flag_label.config(text="")  # No warning
        else:
            distraction_flag_label.config(text=warning_text.strip(), fg=warning_color)  # Display warning message

    # 🔄 Schedule the function to run again in 500ms
    root.after(500, update_text)



def start_processing():
    threading.Thread(target=process_video, daemon=True).start()
    root.after(500, update_text)  # Start updating text

# Start processing when GUI starts
start_processing()

# Start updating frames in the GUI
update_frame()

# Run the Tkinter GUI
root.mainloop()

# Release resources when GUI is closed
cap.release()
cv2.destroyAllWindows()



