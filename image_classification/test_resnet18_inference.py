import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os


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
    model = models.resnet18(weights=None)
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load("fine_tuned_resnet18.pth"))
    model.eval()  # Set model to evaluation mode

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference
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
