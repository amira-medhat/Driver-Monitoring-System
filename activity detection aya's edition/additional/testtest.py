import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define Custom Dataset to load images and labels from CSV
class CustomTestDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        """
        Custom dataset for loading test images and true labels from a CSV file.
        Args:
            image_dir: Directory containing test images.
            csv_file: Path to the CSV file containing image names and their corresponding labels.
            transform: Transformations to apply on the images.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = pd.read_csv(csv_file)  # Read CSV
        self.labels_dict = dict(
            zip(self.labels_df["img"], self.labels_df["classname"])
        )  # Map image filenames to their labels

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]["img"]  # Get image name from CSV
        label = int(self.labels_df.iloc[idx]["classname"][-1])  # Extract numeric class
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to evaluate the model and plot the confusion matrix
def evaluate_and_plot_confusion_matrix(model, test_loader, class_names, device):
    model.eval()  # Ensure model is in evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)  # Get predicted class indices

            # Append true and predicted labels
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix")
    plt.show()

# Define transforms
test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Paths and CSV
test_image_dir = "D:\\grad project\\state_farm_dataset\\imgs\\test"  # Test images directory
csv_file_path = "D:\\grad project\\state_farm_dataset\\driver_imgs_list.csv" # CSV file

# Class labels
classes = {
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

# Load dataset and DataLoader
test_dataset = CustomTestDataset(test_image_dir, csv_file_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(
    torch.load(r"D:\grad project\imgClass_AD\activity detection models\fine_tuned_resnet18_with_how_2.pth", map_location=device)
)
model = model.to(device)

# Evaluate and plot confusion matrix
evaluate_and_plot_confusion_matrix(model, test_loader, classes, device)
