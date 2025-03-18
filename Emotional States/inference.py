import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


#  newwwwwwwwwwwwwwwwwwwwww
# 🔹 Define paths
weights_path = r"C:\Users\Farah\Downloads\mobilenet_emotion.weights.h5"
  # Update with your actual path

# 🔹 Load MobileNetV2 model architecture
def build_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# 🔹 Load the model and weights
model = build_model()
model.load_weights(weights_path)
print("✅ Model loaded successfully!")

# 🔹 Open webcam for real-time inference
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))  # Resize to model input size
    img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    label = "Happy" if prediction[0][0] > 0.4 else "Neutral"

    # Display result
    cv2.putText(frame, f"Emotion: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔹 Release resources
cap.release()
cv2.destroyAllWindows()
