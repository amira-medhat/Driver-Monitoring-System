import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the PyTorch Classification Model for Inference
class PyTorchClassificationModel:
    def __init__(self, model_path, labels, transform, classes, device="cuda"):
        """
        Args:
            model_path: Path to the PyTorch model file (.pth)
            labels: List of label indices
            transform: Transformations to preprocess the input images
            classes: List of class names
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self.load_model(model_path, len(classes))
        self.model.eval()  # Set model to evaluation mode
        self.labels = labels
        self.transform = transform
        self.classes = classes
        self.model = self.model.to(self.device)

    def load_model(self, model_path, num_classes):
        """
        Load the trained PyTorch model.
        """
        model = models.resnet18(weights=None)  # Load ResNet18 architecture
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device), weights_only=True))
        return model

    def preprocess(self, image_path):
        """
        Preprocess the input image: Resize, Normalize, and Convert to Tensor.
        """
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def run_from_image(self, image):
        """
        Perform inference on a single image tensor.
        Args:
            image: Preprocessed image tensor
        Returns:
            Predicted class and probability
        """
        input_tensor = image.unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        probabilities = probabilities.cpu().numpy()
        predicted_label = np.argmax(probabilities)
        predicted_probability = probabilities[predicted_label]
        predicted_class = self.classes[predicted_label]

        return predicted_class, predicted_probability

# Paths and class labels
model_path = "D:\\GRAD_PROJECT\\Driver-Monitoring-System\\Activity Detection\\image_classification\\fine_tuned_resnet18_2.pth"
test_image_dir = "D:\\GRAD_PROJECT\\Driver-Monitoring-System\\Activity Detection\\image_classification\\grey_scaled_imgs"
labels = list(range(0, 10))
classes = [
    "Safe driving",
    "Texting - right",
    "Talking on the phone - right",
    "Texting - left",
    "Talking on the phone - left",
    "Operating the radio",
    "Drinking",
    "Reaching behind",
    "Hair and makeup",
    "Talking to passenger",
]

# Define the transforms for test data
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Add stronger synthetic color variations
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define inverse transforms for visualization (optional)
inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
])

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model = PyTorchClassificationModel(model_path, labels, test_transforms, classes, device=device)

# Perform preprocessing and inference
image_files = os.listdir(test_image_dir)
for image_file in image_files:
    image_path = os.path.join(test_image_dir, image_file)

    # Open the grayscale image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    processed_image = test_transforms(image)

    # Undo normalization for visualization
    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    image_to_show = inv_normalize(processed_image).permute(1, 2, 0).numpy()
    image_to_show = np.clip(image_to_show, 0, 1)

    # Perform inference
    predicted_class, predicted_probability = classification_model.run_from_image(processed_image)

    # Display the processed image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(image_to_show)
    plt.title(f"File: {image_file}\nPredicted Class: {predicted_class}\nProbability: {predicted_probability:.2f}")
    plt.axis("off")
    plt.show()

    # Wait for the user to close the plot window
    input("Press Enter to proceed to the next image...")
