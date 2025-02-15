import torch
import torchvision.models as models

# Load the ResNet50 model pre-trained on ImageNet
resnet18 = models.resnet18(pretrained=True)