import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import copy
import numpy as np
import random
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

SEED = 45
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__=="__main__":
    # Load MobileNetV3-Large pretrained on ImageNet
    # mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
    

    # Load MobileNetV3-Small pretrained on ImageNet
    # mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)

    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=10)
    # print(mobilenet_v3_large)
    
    checkpoint_path = r"C:\Users\Amira\Driver-Monitoring-System\activity detection aya's edition\fine_tuned_mobilenetv3_1.pth"
    mobilenet_v3_large.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    for param in mobilenet_v3_large.features[0:5].parameters():
        param.requires_grad = False

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
    # Load dataset
    train_data = datasets.ImageFolder(root=r"D:\GP_datasets\with_aug\25_2_2025", transform=transform)

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
    # Ensure no in-place modification of the original dataset
    valid_data = copy.deepcopy(valid_data)  
    test_data = copy.deepcopy(test_data)


    print(f"Number of Training examples: {len(train_data)}")
    print(f"Number of Validation examples: {len(valid_data)}")
    print(f"Number of Test examples: {len(test_data)}")

    # Create DataLoader to load batches of images
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = mobilenet_v3_large

    model = model.to(device)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.0001)
    num_epochs = 40  # You can adjust the number of epochs
    model_name = "mobilenetv3"
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
    PATH = "fine_tuned_mobilenetv3_with_aug.pth"
    torch.save(model.state_dict(), PATH)