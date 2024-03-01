import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision import transforms

from myutils import get_device

def load_pretrained_resnet(arch):
    # Dictionary mapping architecture names to their corresponding weights in torchvision
    weights_map = {
        'resnet18': torchvision_models.ResNet18_Weights.DEFAULT,
        'resnet34': torchvision_models.ResNet34_Weights.DEFAULT,
        'resnet50': torchvision_models.ResNet50_Weights.DEFAULT,
    }

    # Check if the specified architecture is supported
    if arch in weights_map:
        # Load the model with the specified weights
        model = torchvision_models.__dict__[arch](weights=weights_map[arch])
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Supported ones are: {list(weights_map.keys())}")
    
    return model

class PointModel(nn.Module):
    def __init__(self, arch, n_points, device=None):
        super(PointModel, self).__init__()
        if device is None:
            self.device = get_device()

        vision_model = load_pretrained_resnet(arch)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Input: b x 512
        # Output: b x args.n_points x 3

        self.n_points = n_points
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.n_points * 3),
            torch.nn.Tanh(),
        )

    def forward(self, images, args):
        images_normalize = self.normalize(images.permute(0, 3, 1, 2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)
        pointclouds_pred = self.decoder(encoded_feat)
        return pointclouds_pred.view(-1, self.n_points, 3)
