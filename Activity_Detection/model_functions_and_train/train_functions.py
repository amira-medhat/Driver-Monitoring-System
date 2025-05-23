import os  # For file system operations (e.g., listing files in directories).
import torch  # PyTorch library for deep learning.
import torch.nn as nn  # Neural network module for building models.
import torch.optim as optim  # Optimizers for training models (e.g., SGD, Adam).
import matplotlib.pyplot as plt  # For visualizing images and results.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # For working with datasets and dataloaders.
from torchvision import (
    datasets,
    models,
    transforms,
)  # Pre-trained models (ResNet18), and image preprocessing utilities.
from PIL import Image  # For working with image files.


def plot_images_RGB(images, n_cols=5):
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


def plot_images_RGB_without_labels(images, n_cols=5):
    """
    Plots a grid of RGB images without labels.

    :param images: List of images (tensors or PIL images).
    :param n_cols: Number of columns in the grid. Default is 5.
    """
    n_rows = (len(images) + n_cols - 1) // n_cols  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()  # Flatten axes for easy iteration

    for ax, image in zip(axes, images):
        if isinstance(image, torch.Tensor):  # If the image is a Tensor
            image = image.permute(
                1, 2, 0
            ).numpy()  # Convert CHW to HWC for visualization

        ax.imshow(image)
        ax.axis("off")  # Turn off axis labels

    # Hide any unused axes
    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_images_grayscale(images, n_cols=5):
    """
    Plots a grid of grayscale images with optional class labels.

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
            if image.ndimension() == 3:  # Handle case where image has a single channel
                image = image.squeeze(0)  # Remove channel dimension
            image = image.numpy()  # Convert tensor to numpy array

        ax.imshow(image, cmap="gray")  # Display as grayscale
        ax.axis("off")  # Turn off axis labels
        ax.set_title(class_name, fontsize=8)  # Add class name as title

    # Hide any unused axes
    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_images_grayscale_without_labels(images, n_cols=5):
    """
    Plots a grid of grayscale images without labels.

    :param images: List of images (tensors or PIL images).
    :param n_cols: Number of columns in the grid. Default is 5.
    """
    n_rows = (len(images) + n_cols - 1) // n_cols  # Calculate the number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()  # Flatten axes for easy iteration

    for ax, image in zip(axes, images):
        if isinstance(image, torch.Tensor):  # If the image is a Tensor
            if image.ndimension() == 3:  # Handle case where image has a single channel
                image = image.squeeze(0)  # Remove channel dimension
            image = image.numpy()  # Convert tensor to numpy array

        ax.imshow(image, cmap="gray")  # Display as grayscale
        ax.axis("off")  # Turn off axis labels

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
    train_iterator,
    valid_iterator,
    optimizer,
    loss_criterion,
    device,
    epochs,
    checkpoint_path="model_checkpoint.pth",
):
    """
    Train and validate the model for a given number of epochs, while tracking performance statistics.
    Save checkpoint after each epoch to allow resuming training if interrupted.

    :param model: The PyTorch model.
    :param model_name: The name of the model (for tracking).
    :param train_iterator: DataLoader for the training set.
    :param valid_iterator: DataLoader for the validation set.
    :param optimizer: The optimizer to use (e.g., Adam).
    :param loss_criterion: The loss function (e.g., CrossEntropyLoss).
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param epochs: Number of epochs to train the model.
    :param checkpoint_path: Path to save checkpoint file.

    :return: Dictionary containing training and validation loss/accuracy statistics for each epoch.
    """
    # Initialize training stats
    train_stats = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Try to load checkpoint if it exists
    start_epoch = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        train_stats = checkpoint["train_stats"]
        print(f"Resuming training from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, epochs):
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
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_stats": train_stats,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

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


# Helper function to fetch images and labels from the DataLoader
def get_images_from_loader(data_loader, n_samples=10):
    """
    Fetches a batch of images and corresponding labels from a DataLoader.

    :param data_loader: DataLoader object.
    :param n_samples: Number of images to fetch.
    :return: List of tuples (image, class_name).
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    class_names = [
        str(label.item()) for label in labels[:n_samples]
    ]  # Convert labels to strings
    images = images[:n_samples]  # Select the first n_samples images
    return [
        (image.squeeze(0), class_name) for image, class_name in zip(images, class_names)
    ]


def get_images_from_loader_without_labels(data_loader, n_samples=5):
    """
    Function to get sample images from a DataLoader that does not contain labels for visualization.
    Args:
        data_loader: DataLoader object containing the test images.
        n_samples: Number of samples to retrieve.
    Returns:
        A list of sample images.
    """
    images = []

    # Iterate through the DataLoader
    for i, (image_batch, _) in enumerate(data_loader):
        if i >= n_samples:
            break  # Stop after n_samples
        images.append(image_batch)

    # Flatten the list of image batches
    images = torch.cat(images, dim=0)[:n_samples]  # Limit to n_samples
    return images


def plot_confusion_matrix(model, test_loader, device, classes):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes.values())
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
