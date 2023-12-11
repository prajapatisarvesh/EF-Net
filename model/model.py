from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from base.base_model import BaseModel
from torchvision.models import densenet169
from torchvision.models.feature_extraction import create_feature_extractor


class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self,out_channels=47,features=[64, 128, 256, 512]):
        super(UNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3],features[3]*2)
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
    
    
class EndToEndFrictionEstimation(BaseModel):
    def __init__(self,block_indices=[0,1]):
        super().__init__()
        self.densenet = densenet169(pretrained=True)
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        return_nodes = {
            "features.transition1.pool": "features1",
            "features.denseblock2.denselayer12.conv2": "features2",
        }
        self.model = create_feature_extractor(self.densenet, return_nodes=return_nodes)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=(3,3), stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=24*24*9, out_features=1550)
        self.fc2 = nn.Linear(in_features=1550, out_features=1550)


    def forward(self, x):
        x = self.model(x)
        x = torch.cat((x['features1'], x['features2']), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 24*24*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x