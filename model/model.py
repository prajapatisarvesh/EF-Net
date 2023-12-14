'''
LAST UPDATE: 2023.12.10
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


'''
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from base.base_model import BaseModel
from torchvision.models import densenet169
from torchvision.models.feature_extraction import create_feature_extractor


class EndToEndFrictionEstimation(BaseModel):
    '''
    EF-Net: End-to-End Friction Estimation

    Authors -- Sarvesh, Abhinav, Rupesh

    Basic model architecture:
        1. DenseNet169 pretrained on ImageNet
        2. Taking the output from transition1.pool and denseblock2.denselayer12.conv2
        3. Concatenating the two outputs
        4. Passing through two convolutional layers and reducing the features
        5. Passing through two fully connected layers for estimating the spectral value of material
    '''
    def __init__(self,block_indices=[0,1]):
        '''
        Constructor for the model
        Pretrained DenseNet defined here
        Features are extracted from the two layers using the create_feature_extractor function
        '''
        super().__init__()
        # Pretrained Densenet
        self.densenet = densenet169(pretrained=True)
        # Don't train DenseNet (freeze gradients)
        for param in self.densenet.parameters():
            param.requires_grad = False
        # Features to be extracted from the DenseNet
        return_nodes = {
            "features.transition1.pool": "features1",
            "features.denseblock2.denselayer12.conv2": "features2",
        }
        # Define model
        self.model = create_feature_extractor(self.densenet, return_nodes=return_nodes)
        # First conv layer that takes in concatenated features from densenet
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=(3,3), stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=24*24*9, out_features=1550)
        # last layer for getting spectral data
        self.fc2 = nn.Linear(in_features=1550, out_features=1550)


    def forward(self, x):
        x = self.model(x)
        # concat features
        x = torch.cat((x['features1'], x['features2']), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 24*24*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # return predicted features
        return x

class unet_encoder(nn.Module):
    '''
    Basic Encoder block for the UNet with Conv->BatchNorm->ReLU->Conv->BatchNorm->ReLU
    Can be used for expansive and contractive paths
    '''
    def __init__(self,in_channels, out_channels):
        super(unet_encoder,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    '''
    Unet Model
    
    Out channels = 1 for regression and num_classses for segmentation/classification
    for classification, use sigmoid and argmax on output
    
    Reference -- 
    [1]O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” CoRR, vol. abs/1505.04597, 2015, [Online]. Available: http://arxiv.org/abs/1505.04597 
    '''
    def __init__(self,out_channels=60,features=[64, 128, 256, 512]):
        super(UNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = unet_encoder(in_channels=3,out_channels=features[0])
        self.conv2 = unet_encoder(in_channels=features[0],out_channels=features[1])
        self.conv3 = unet_encoder(in_channels=features[1],out_channels=features[2])
        self.conv4 = unet_encoder(in_channels=features[2],out_channels=features[3])
        self.conv5 = unet_encoder(in_channels=features[3]*2,out_channels=features[3])
        self.conv6 = unet_encoder(in_channels=features[3],out_channels=features[2])
        self.conv7 = unet_encoder(in_channels=features[2],out_channels=features[1])
        self.conv8 = unet_encoder(in_channels=features[1],out_channels=features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = unet_encoder(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x
    
'''
Taken from Assignment 1 CS7180
Authors -- Sarvesh, Abhinav, Rupesh
'''
class SRCNN(BaseModel):
    def __init__(self):
        super().__init__()
        ### First conv2d layer, which takes in bicubic interpolated image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        ### Non linear mapping
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        ### Output for SRCNN
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    '''
    Forward function for tensors
    '''
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
    
