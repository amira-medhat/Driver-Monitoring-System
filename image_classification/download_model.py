import torch
import torchvision.models as models

# Load the ResNet50 model pre-trained on ImageNet
resnet50 = models.resnet50(pretrained=True)
