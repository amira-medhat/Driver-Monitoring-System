'''
1. Imports
'''

import torch   #PyTorch library for deep learning.
import torch.nn as nn       #Neural network module for building models.
import torch.optim as optim     #Optimizers for training models (e.g., SGD, Adam).
import matplotlib.pyplot as plt      #For visualizing images and results.
from torch.utils.data import DataLoader, Dataset    #For working with datasets and dataloaders.
from torchvision import datasets, models, transforms    #Pre-trained models (ResNet18), and image preprocessing utilities.
from PIL import Image   #For working with image files.
import os   #For file system operations (e.g., listing files in directories).
#from matplotlib import pyplot as plt



if __name__ == "__main__":

    '''
    2. Custom Dataset Class (CustomTestDataset)
    '''
    class CustomTestDataset(Dataset):  
        '''
        This class is designed to load images from a directory for inference.
        Unlike training datasets, test datasets in this case have no subdirectories for classes.
        This custom class simplifies loading and preprocessing test images.
        '''
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

        def __getitem__(self, idx):     #Loads an image at a specific index, applies transformations (e.g., resizing, normalization), and returns it.
            img_name = os.path.join(self.image_dir, self.image_files[idx])
            image = Image.open(img_name)  # Open the image
            if self.transform:
                image = self.transform(
                    image
                )  # Apply any transformations (e.g., resizing, normalization)
            return image


    '''
    3. Image Preprocessing
    '''
    # Define the transformations: Resize, Convert to Tensor, and Normalize
    transform = transforms.Compose(
        '''
        These transformations ensure the images are consistent with the format expected by ResNet18,
        which is pre-trained on ImageNet.
        '''
        [
            transforms.Resize(224),  # Resize image to (224, 224)
            transforms.CenterCrop(224),  # Crop the center of the image of size (224, 224).
            transforms.ToTensor(),  #Converts the image to a PyTorch tensor (shape [C, H, W]).
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize based on ImageNet statistics(mean ,std)
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


    '''
   4. Loading Datasets
    '''
    # Define paths to your dataset(train set / test set)
    train_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\train"
    test_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\test"

    # Load the dataset
    '''
    When your dataset is organized with class labels in separate subdirectories.
    Ideal for image classification problems where the class name is derived from the folder name.
    '''
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Create the dataset for test data
    '''
    When the dataset doesn't have class subdirectories, such as a test set where you only have images without predefined class labels.
    When you need to handle the data in a custom way, like loading images from a CSV file, applying custom transformations, or working with images in a non-standard format.
    '''
    test_dataset = CustomTestDataset(test_dir, transform=transform)

    # Create DataLoader to load batches of images
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    '''
    5. Model Customization
    '''
    # Load ResNet18 with pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final layer to match your dataset's number of classes
    num_classes = len(train_dataset.classes)  # Number of classes in your dataset
    model.fc = nn.Linear(
        model.fc.in_features, num_classes
    )  # The final fully connected layer (model.fc) is replaced with a new layer to match the number of classes in the training dataset.

    # Freeze all layers except the final layer (fc) to prevent updates during training. 
    # This reduces training time and focuses on fine-tuning the last layer.
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers.

    # Unfreeze the final layer to fine-tune it
    for param in model.fc.parameters():
        param.requires_grad = True


    '''
    6. Training the Model
    '''
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()   #suitable for multi-class classification.

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

    torch.save(model.state_dict(), "fine_tuned_resnet18.pth")   #The model's state is saved to a file (fine_tuned_resnet18.pth) after training.
