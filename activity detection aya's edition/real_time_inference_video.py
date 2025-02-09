import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys

# Define ResNet-18 Model for Activity Detection
class CustomResNet18(nn.Module):
    def __init__(self, model_path, labels, classes, device="CPU"):
        super(CustomResNet18, self).__init__()
        self.device = device
        self.model = self.load_model(model_path, len(classes))
        self.model.eval()
        self.labels = labels
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        return model

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11

model_path = r"D:\grad project\imgClass_AD\activity detection models\fine_tuned_resnet18_with_how_2.pth"
labels = list(range(0, 11))
class_labels  = {
    0: "Safe driving",
    1: "Texting(right hand)",
    3: "Talking on the phone (right hand)",
    4: "Texting (left hand)",
    5: "Talking on the phone (left hand)",
    6: "Operating the radio",
    7: "Drinking",
    8: "Reaching behind",
    9: "Hair and makeup",
    10: "Talking to passenger(s)",
    2: "Hands off Wheel",
}

model = CustomResNet18(model_path, labels, class_labels, device=device).model

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_clahe(image):
    """Applies CLAHE to normalize brightness variations in different lighting conditions."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Apply CLAHE to L-channel
    l = clahe.apply(l)
    
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convert back to RGB


class ActivityDetection(QMainWindow):
    def __init__(self, video_path=None):
        super().__init__()

        self.alert_text = ""
        self.driver_state = ""

        self.setWindowTitle("Driver Activity Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)

        self.update_info()

        # Open video file
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            print("No video selected. Exiting.")
            sys.exit()

        time.sleep(1.000)

        # Multi-Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

    def update_info(self):
        """Update activity info on UI."""
        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333;'>"
            f"<h2 style='text-align: center; color: #4CAF50;'>Driver Activity Detection</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"
            f"<p><b> Current Driver State:</b> {self.driver_state}</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)
    '''
    def capture_frames(self):
        """Capture video frames and store them in the queue."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                print("End of video. Exiting.")
                self.stop_event.set()
                break
    '''

    def capture_frames(self):
        """Capture video frames and store them in the queue while maintaining real-time speed."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # Get original video FPS
        frame_time = 1 / fps  # Calculate time per frame

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
                time.sleep(frame_time)  # Add delay to match FPS
            else:
                print("End of video. Exiting.")
                self.stop_event.set()
                break


    def process_frames(self):
        """Process frames for real-time activity detection."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                try:
                    self.driver_state = self.predict_activity(frame)
                except Exception as e:
                    print(f"Error in activity prediction: {e}")
                    self.driver_state = "Unknown"

                if self.driver_state == "Hands off Wheel":
                    self.alert_text = "Alert: Hands off the wheel detected!"
                elif self.driver_state == "Reaching behind":
                    self.alert_text = "Alert: Pay attention to the road!"

                self.update_info()
                self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def predict_activity(self, frame):
        """Predict driver activity using ResNet-18."""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply CLAHE for illumination robustness
            img_rgb = apply_clahe(img_rgb)

            img = Image.fromarray(img_rgb)
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                label = class_labels[predicted.item()]

            return label

        except Exception as e:
            print(f"Error in activity prediction: {e}")
            return "Unknown"

    def display_frame(self, frame):
        """Display frame in PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ActivityDetection()
    window.show()
    sys.exit(app.exec_())

'''
import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys

# Define ResNet-18 Model for Activity Detection
class CustomResNet18(nn.Module):
    def __init__(self, model_path, labels, classes, device="CPU"):
        """
        Args:
            model_path: Path to the PyTorch model file (.pth)
            labels: List of label indices
            classes: List of class names
            device: 'cuda' or 'cpu'
        """
        super(CustomResNet18, self).__init__()  # Initialize the parent class
        self.device = device  # Initialize device first
        self.model = self.load_model(model_path, len(classes))  # Load model
        self.model.eval()  # Set model to evaluation mode
        self.labels = labels
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        """
        Load the trained PyTorch model.
        """
        model = models.resnet18(weights=None)  # Load ResNet18 architecture
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))  # Load weights
        return model

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11

# Load model weights
model_path = r"D:\grad project\imgClass_AD\activity detection models\fine_tuned_resnet18_with_how_2.pth"
labels = list(range(0, 10))
# Define class labels (modify based on dataset)
class_labels = {
    0: "Safe driving",
    1: "Texting(right hand)",
    3: "Talking on the phone (right hand)",
    4: "Texting (left hand)",
    5: "Talking on the phone (left hand)",
    6: "Operating the radio",
    7: "Drinking",
    8: "Reaching behind",
    9: "Hair and makeup",
    10: "Talking to passenger(s)",
    2: "Hands off Wheel",
}

model = CustomResNet18(model_path, labels, class_labels, device=device).model

# Image Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.CenterCrop(224),  # Crop the center
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet stats)
])


class ActivityDetection(QMainWindow):
    def __init__(self, video_path=None):
        super().__init__()

        # Store current states
        self.alert_text = ""
        self.driver_state = ""

        self.setWindowTitle("Driver Activity Detection")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)

        self.update_info()

        # Open video file
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            print("No video selected. Exiting.")
            sys.exit()

        time.sleep(1.000)

        # Using Multi-Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

    def update_info(self):
        """Update activity info on UI."""
        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333;'>"
            f"<h2 style='text-align: center; color: #4CAF50;'>Driver Activity Detection</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"  # Display alert if exists
            f"<p><b> Current Driver State:</b> {self.driver_state}</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    def capture_frames(self):
        """Capture video frames and store them in the queue."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                print("End of video. Exiting.")
                self.stop_event.set()
                break

    def process_frames(self):
        """Process frames for real-time activity detection."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                try:
                    self.driver_state = self.predict_activity(frame)  # Predict activity
                except Exception as e:
                    print(f"Error in activity prediction: {e}")
                    self.driver_state = "Unknown"

                if self.driver_state == "Hands off Wheel":
                    self.alert_text = "Alert: Hands off the wheel detected!"
                    #self.play_sound_in_thread()
                elif self.driver_state == "Reaching behind":
                    self.alert_text = "Alert: Pay attention to the road!"
                    #self.play_sound_in_thread()

                # Overlay result on the frame
                #self.display_prediction(frame, self.driver_state)
                self.update_info()
                self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def predict_activity(self, frame):
        """Predict driver activity using ResNet-18."""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, but Resnet18 requires RGB, so conversion is needed.
            img = Image.fromarray(img_rgb)

            # Preprocess
            img = transform(img).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                label = class_labels[predicted.item()]

            return label

        except Exception as e:
            print(f"Error in activity prediction: {e}")
            return "Unknown"

    def display_frame(self, frame):
        """Display frame in PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ActivityDetection()
    window.show()
    sys.exit(app.exec_())
'''
