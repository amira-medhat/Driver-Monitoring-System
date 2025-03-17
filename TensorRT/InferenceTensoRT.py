import multiprocessing
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from torchvision import transforms
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Global variables
engine_path = '/home/farouk/Public/Workspace/model.engine'
class_labels = {
    0: "Safe driving",
    1: "Texting(right hand)",
    2: "Talking on the phone (right hand)",
    3: "Texting (left hand)",
    4: "Talking on the phone (left hand)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger(s)",
}

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_activity(frame, context, inputs, outputs, bindings, stream):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(Image.fromarray(img_rgb)).unsqueeze(0).numpy()

    np.copyto(inputs[0][0], img.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    logits = outputs[0][0]
    
    # Apply softmax using NumPy (no torch dependency)
    probabilities = softmax(logits)

    top3_indices = probabilities.argsort()[-3:][::-1]
    top3_labels = [class_labels[idx] for idx in top3_indices]
    top3_confidences = [round(probabilities[idx] * 100, 2) for idx in top3_indices]

    return list(zip(top3_labels, top3_confidences))


def capture_frames(frame_queue, stop_event, video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 1 / 30
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret and frame_queue.qsize() < 2:
            frame_queue.put(frame)
            time.sleep(frame_time)
        elif not ret:
            stop_event.set()
    cap.release()

def process_frames(frame_queue, result_queue, stop_event):
    engine = load_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
            predictions = predict_activity(frame, context, inputs, outputs, bindings, stream)
            result_queue.put((frame, predictions))
        except Exception as e:
            print(f"Error: {e}")

# GUI class remains unchanged (ActivityDetection)

class ActivityDetection(QMainWindow):
    def __init__(self, frame_queue, result_queue, stop_event, video_path):
        super().__init__()

        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

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
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 18px;")
        self.layout.addWidget(self.info_label)

        self.update_info()

        self.timer = self.startTimer(30)  # 30ms update interval

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

    def timerEvent(self, event):
        """Update UI with processed frame results."""
        try:
            frame, predictions = self.result_queue.get_nowait()

            #print(f"Updating UI with: {predictions}")  # Debugging print

            if predictions:
                self.driver_state = predictions[0][0]  # Set the highest-confidence prediction
                self.alert_text = f"<b>Alert:</b> {self.driver_state} detected!" if self.driver_state != "Safe driving" else ""

                confidence_display = "<br>".join([f"{cls}: {conf}%" for cls, conf in predictions])

                self.info_label.setText(
                    f"<b>Current Driver State:</b> {self.driver_state}<br>"
                    f"<b>Top Predictions:</b><br>{confidence_display}"
                )

                self.display_frame(frame)  # Ensure the frame is updated

        except Exception as e:
            pass
            # print(f"UI Update Error: {e}")


    def display_frame(self, frame):
        """Display frame in PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

        
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)

    video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    if not video_path:
        print("No video selected. Exiting.")
        sys.exit()

    frame_queue = multiprocessing.Queue(maxsize=2)
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue, stop_event, video_path))
    inference_process = multiprocessing.Process(target=process_frames, args=(frame_queue, result_queue, stop_event))

    capture_process.start()
    inference_process.start()

    window = ActivityDetection(frame_queue, result_queue, stop_event, video_path)
    window.show()
    sys.exit(app.exec_())