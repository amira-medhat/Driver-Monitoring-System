import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":

    class CustomTestDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            """
            Custom Dataset for test data with no class subdirectories.
            Args:
                image_dir (str): Path to the directory with test images.
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.image_dir = image_dir
            self.image_files = os.listdir(image_dir)  # List all files in the directory
            self.transform = transform

        def __len__(self):
            return len(self.image_files)  # Return the number of images

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_name)  # Open the image
            if self.transform:
                image = self.transform(
                    image
                )  # Apply any transformations (e.g., resizing, normalization)
            return image

    # Define the transformations: Resize, Convert to Tensor, and Normalize
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # Resize image to (224, 224)
            transforms.CenterCrop(224),  # Crop the center of the image
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize based on ImageNet stats
        ]
    )
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(224),
    #         transforms.RandomHorizontalFlip(),  # Flip images horizontally
    #         transforms.RandomRotation(10),  # Rotate images slightly
    #         transforms.RandomResizedCrop(224),  # Randomly crop and resize
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # Define paths to your dataset
    train_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\train"
    test_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\test"

    # Load the dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Create the dataset for test data
    test_dataset = CustomTestDataset(test_dir, transform=transform)

    # Create DataLoader to load batches of images
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load ResNet18 with pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final layer to match your dataset's number of classes
    num_classes = len(train_dataset.classes)  # Number of classes in your dataset
    model.fc = nn.Linear(
        model.fc.in_features, num_classes
    )  # Change the last fully connected layer

    # Freeze all layers except the final layer (fc)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers

    # Unfreeze the final layer to fine-tune it
    for param in model.fc.parameters():
        param.requires_grad = True

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer (you can adjust learning rate)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 50  # You can adjust the number of epochs

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    torch.save(model.state_dict(), "fine_tuned_resnet18.pth")
