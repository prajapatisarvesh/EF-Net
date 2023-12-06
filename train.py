import torch
import torch.nn as nn
import torchvision


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device=device)
    print(model)