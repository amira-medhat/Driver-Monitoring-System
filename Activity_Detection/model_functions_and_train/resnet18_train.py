"""
1. Imports
"""

import os  # For file system operations (e.g., listing files in directories).
import copy
import random
import torch  # PyTorch library for deep learning.
import torch.nn as nn  # Neural network module for building models.
import torch.optim as optim  # Optimizers for training models (e.g., SGD, Adam).
import matplotlib.pyplot as plt  # For visualizing images and results.
import pandas as pd
import numpy as np
import pdb
import seaborn as sns
from IPython.display import display
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torchvision import (
    datasets,
    models,
    transforms,
)
from PIL import Image  # For working with image files.
from train_functions import (
    plot_images_RGB,
    evaluate,
    fit_model,
    count_parameters,
    plot_training_statistics,
    get_images_from_loader,
    plot_images_grayscale,
    plot_confusion_matrix,
)

torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)


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

"""
3. Image Preprocessing
"""
# labels = pd.read_csv(
#     "D:\\grad project\\state_farm_dataset\\driver_imgs_list.csv"
# )

# display(labels.head())

# Define paths to your dataset(train set / test set)
train_dir = r"D:\GP_datasets\with_aug\25_2_2025"

# num_training_examples = 0
# for fol in os.listdir(train_dir):
#     num_training_examples += len(os.listdir(os.path.join(train_dir, fol)))

# # assert num_training_examples == len(labels)

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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)
train_data = datasets.ImageFolder(root=train_dir, transform=transform)

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


# Apply the new transforms to the datasets

valid_data = copy.deepcopy(
    valid_data
)  # Ensure no in-place modification of the original dataset
test_data = copy.deepcopy(test_data)


print(f"Number of Training examples: {len(train_data)}")
print(f"Number of Validation examples: {len(valid_data)}")
print(f"Number of Test examples: {len(test_data)}")

"""
4. Loading Datasets
"""

# Load the dataset

# Create DataLoader to load batches of images
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)


# # Fetch a sample from the training data
# print("Training Data Samples:")
# train_images = get_images_from_loader(train_loader, n_samples=5)
# plot_images_RGB(train_images)

# # Fetch a sample from the validation data
# print("Validation Data Samples:")
# valid_images = get_images_from_loader(valid_loader, n_samples=5)
# plot_images_RGB(valid_images)

# # Fetch a sample from the test data
# print("Test Data Samples:")
# test_images = get_images_from_loader(test_loader, n_samples=5)
# plot_images_RGB(test_images)


"""
5. Model Customization
"""
# Load ResNet18 with pre-trained weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# print(model)

# Modify the final layer to match your dataset's number of classes
# for name, param in model.named_parameters():
#     if "bn" not in name:
#         param.requires_grad = False


for name, param in model.named_parameters():
    if name.startswith("conv1") or name.startswith("layer1") or name.startswith("layer2") or name.startswith("layer3"):
        param.requires_grad = False  # Freeze these layers

model.fc = nn.Linear(model.fc.in_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
for images, labels in valid_loader:
    images, labels = images.to(device), labels.to(device)
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

"""
6. Training the Model
"""

# Define the loss function
criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification.

# Use Adam optimizer (you can adjust learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 50  # You can adjust the number of epochs
model_name = "ResNet18"
print(f"The model has {count_parameters(model):,} trainable parameters")
train_stats = fit_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
)
plot_training_statistics(train_stats, model_name)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
plot_confusion_matrix(model, test_loader, device, classes)
PATH = "fine_tuned_resnet18_with_aug_2.pth"
torch.save(model.state_dict(), PATH)
