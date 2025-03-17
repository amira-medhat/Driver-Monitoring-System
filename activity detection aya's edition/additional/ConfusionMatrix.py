import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Paths
split_dir = "D:\\grad project\\split_dataset"
test_dir = os.path.join(split_dir, "train")
model_path = r"D:\grad project\imgClass_AD\activity detection models\fine_tuned_resnet18_with_how_2.pth"  # Path to the trained model

# Class mapping
classes = {
    "c0": "Safe driving",
    "c1": "Texting (right hand)",
    "c2": "Talking on the phone (right hand)",
    "c3": "Texting (left hand)",
    "c4": "Talking on the phone (left hand)",
    "c5": "Operating the radio",
    "c6": "Drinking",
    "c7": "Reaching behind",
    "c8": "Hair and makeup",
    "c9": "Talking to passenger(s)",
    "c10": "Hands off Wheel",
}

# Image transformations (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
test_data = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

actual_class_to_idx = test_data.class_to_idx
print(f"Class-to-index mapping: {actual_class_to_idx}")

classes = {v: classes[k] for k, v in actual_class_to_idx.items()}
print(f"Index-to-class mapping: {classes}")


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import models
import torch.nn as nn

# Define the model and load the weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 11)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

model = model.to(device)


model.eval()

# Initialize lists for predictions and ground truth
all_preds = []
all_labels = []

# Perform inference on the test set
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Get predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(classes.values()))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()
