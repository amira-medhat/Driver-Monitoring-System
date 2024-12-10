import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os

# EL REPO SHAGHAAAAAAAL!!!!

class CustomTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # Return the image and its filename


if __name__ == "__main__":
    # Define the transformations for test data
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Path to the test dataset
    test_dir = "C:\\Users\\Amira\\state-farm-distracted-driver-detection\\imgs\\test"

    # Create the test dataset and DataLoader
    test_dataset = CustomTestDataset(test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Define class names
    class_names = [
        "safe driving",
        "texting - right",
        "talking on the phone - right",
        "texting - left",
        "talking on the phone - left",
        "operating the radio",
        "drinking",
        "reaching behind",
        "hair and makeup",
        "talking to passenger",
    ]

    # Load the model architecture
    '''
    Load Model Architecture:
    A ResNet18 architecture is loaded.
    The fully connected layer (model.fc) is replaced to output predictions for the 10 classes (matching the dataset).
    '''
    model = models.resnet18(weights=None)
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    '''
    Load Pre-trained Weights:
    The model's weights are loaded from the file fine_tuned_resnet18.pth, which was saved after fine-tuning on the dataset.
    '''
    model.load_state_dict(torch.load("fine_tuned_resnet18.pth"))
    model.eval()  # Set model to evaluation mode

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference
    '''
    Test images are normalized for model input, but for visualization, normalization is reversed using inv_transform.
    '''
    inv_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            '''
            outputs(row x col 'image x class') =
                tensor([[2.5, 1.0, -0.5, 3.0],  # Image 1 logits
                        [0.8, 2.2, 0.1, -1.5],  # Image 2 logits
            
            outputs: The tensor containing logits.
            1: This specifies the dimension along which to compute the maximum.
            1 means find the maximum across the columns (i.e., across the classes for each image).
            '''

            for i in range(images.size(0)):
                # Revert normalization for visualization
                image = inv_transform(images[i].cpu())
                image = image.permute(1, 2, 0).numpy()

                # Display the image with prediction
                plt.imshow(image)
                plt.title(
                    f"File: {filenames[i]}, Predicted: {class_names[predicted[i]]}"
                )
                plt.axis("off")
                plt.show()
