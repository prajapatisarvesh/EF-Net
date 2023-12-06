import torch
import torch.nn as nn
import torchvision
from data_loader.data_loaders import AppleMLDMSLoader
import os

if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device=device)
    # print(model)
    data = AppleMLDMSLoader(csv_file='./data/dms-dataset-final/train.csv', root_dir=os.getcwd())
    # for i in range(10):
    #     print(data.__getitem__(i)['label'].shape)