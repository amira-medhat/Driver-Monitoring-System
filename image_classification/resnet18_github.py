import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import copy
import time
from math import exp
import random
import os
from PIL import Image
import seaborn as sns
from IPython.display import display

import torch
import torchvision
from torch import nn, optim
from torchvision import models
from torch.functional import F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable

sns.set_style("whitegrid")
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["font.size"] = 12
SEED = 47


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def plot_images(images, n_cols=5):
    """
    Plots a grid of images with optional class labels.

    :param images: List of tuples (image, class_name), where:
                   - image is a tensor or PIL image.
                   - class_name is a string representing the class label.
    :param n_cols: Number of columns in the grid. Default is 5.
    """
    n_rows = (len(images) + n_cols - 1) // n_cols  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()  # Flatten axes for easy iteration

    for ax, (image, class_name) in zip(axes, images):
        if isinstance(image, torch.Tensor):  # If the image is a Tensor
            image = image.permute(
                1, 2, 0
            ).numpy()  # Convert CHW to HWC for visualization

        ax.imshow(image)
        ax.axis("off")  # Turn off axis labels
        ax.set_title(class_name, fontsize=8)  # Add class name as title

    # Hide any unused axes
    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    :param model: PyTorch model
    :return: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_model(
    model,
    model_name,
    train_iterator,
    valid_iterator,
    optimizer,
    loss_criterion,
    device,
    epochs,
):
    """
    Train and validate the model for a given number of epochs, while tracking performance statistics.

    :param model: The PyTorch model.
    :param model_name: The name of the model (for tracking).
    :param train_iterator: DataLoader for the training set.
    :param valid_iterator: DataLoader for the validation set.
    :param optimizer: The optimizer to use (e.g., Adam).
    :param loss_criterion: The loss function (e.g., CrossEntropyLoss).
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param epochs: Number of epochs to train the model.

    :return: Dictionary containing training and validation loss/accuracy statistics for each epoch.
    """
    train_stats = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_iterator:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track the loss and accuracy
            running_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy for this epoch
        avg_train_loss = running_train_loss / len(train_iterator)
        train_accuracy = (correct_train / total_train) * 100
        train_stats["train_loss"].append(avg_train_loss)
        train_stats["train_accuracy"].append(train_accuracy)

        # Validation step
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for images, labels in valid_iterator:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_criterion(outputs, labels)

                # Track the loss and accuracy for validation
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy for this epoch
        avg_val_loss = running_val_loss / len(valid_iterator)
        val_accuracy = (correct_val / total_val) * 100
        train_stats["val_loss"].append(avg_val_loss)
        train_stats["val_accuracy"].append(val_accuracy)

        # Print statistics for each epoch
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

    return train_stats


def plot_training_statistics(stats, model_name):
    """
    Plot training statistics such as loss and accuracy.
    :param stats: Dictionary containing 'train_loss', 'train_accuracy', 'val_loss', and 'val_accuracy'.
    :param model_name: The model name for the plot title.
    """
    epochs = range(1, len(stats["train_loss"]) + 1)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats["train_loss"], label="Training Loss")
    plt.plot(epochs, stats["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, stats["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, stats["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate(model, test_iterator, loss_criterion, device):
    """
    Evaluate the model on the test set.
    :param model: The trained model.
    :param test_iterator: DataLoader for the test set.
    :param loss_criterion: Loss function.
    :param device: Device to run the evaluation on ('cpu' or 'cuda').
    :return: Test loss and accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_iterator:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_iterator)
    test_acc = (correct / total) * 100
    return test_loss, test_acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "C:\\Users\\Amira\\Driver-Monitoring-System"
labels = pd.read_csv(
    "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\driver_imgs_list.csv"
)
display(labels.head())

train_img_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\train"
test_img_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\test"

num_training_examples = 0
for fol in os.listdir(train_img_dir):
    num_training_examples += len(os.listdir(os.path.join(train_img_dir, fol)))

assert num_training_examples == len(labels)

classes = {
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

train_data = torchvision.datasets.ImageFolder(root=train_img_dir)
labelss = labels.classname.map(train_data.class_to_idx)
N_IMAGES = 20
images = [
    (image, classes[label])
    for image, label in [
        train_data[i] for i in random.sample(range(len(train_data)), N_IMAGES)
    ]
]
plot_images(images)

RATIO = 0.8

n_train_examples = int(len(train_data) * RATIO)
n_Test_Valid_examples = len(train_data) - n_train_examples
n_valid_examples = int(n_Test_Valid_examples / 2)
n_Test_examples = n_Test_Valid_examples - n_valid_examples

train_data, Test_valid_data = torch.utils.data.random_split(
    train_data, [n_train_examples, n_Test_Valid_examples]
)

valid_data, test_data = torch.utils.data.random_split(
    Test_valid_data, [n_valid_examples, n_Test_examples]
)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)
validation_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

train_data.dataset.transform = train_transforms
valid_data = copy.deepcopy(valid_data)
test_data = copy.deepcopy(test_data)
valid_data.dataset.transform = validation_transforms

print(f"Number of Training examples: {len(train_data)}")
print(f"Number of Validation examples: {len(valid_data)}")
print(f"Number of Test examples: {len(test_data)}")

BATCH_SIZE = 256

train_iterator = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE * 2)

test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE * 2)

model = models.resnet18(pretrained=True)

for name, param in model.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)
loss_criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 20
model_name = "ResNet18"
print(f"The model has {count_parameters(model):,} trainable parameters")
train_stats_ResNet18 = fit_model(
    model,
    model_name,
    train_iterator,
    valid_iterator,
    optimizer,
    loss_criterion,
    device,
    epochs,
)
plot_training_statistics(train_stats_ResNet18, model_name)
test_loss, test_acc = evaluate(model, test_iterator, loss_criterion, device)
test_loss, test_acc
PATH = "resnet18_github.pth"
torch.save(model.state_dict(), PATH)
