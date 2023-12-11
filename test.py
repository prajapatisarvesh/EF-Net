import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from data_loader.data_loaders import AppleMLDMSLoader, VastDataLoader
import os
from datetime import datetime
from typing import Dict, List, Tuple
import torch.nn.functional as F
import torch
import torchvision
from tqdm import trange, tqdm
from model.model import UNet, SRCNN, EndToEndFrictionEstimation
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
import argparse







def test_srcnn_regression():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), mapped=True, transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=True)
    # model = SRCNN().to(device)
    model = UNet(out_channels=1).to(device)
    model.load_state_dict(torch.load('checkpoints/epoch_unetr_112.pt'))
    # model.load_state_dict(torch.load('checkpoints/epoch_srcnn_99.pt'))
    for x,y in test:
        x = x.to(device)
        fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds = model(x).to('cpu').detach()
        print(preds.min(), preds.max(), y.min(), y.max(), preds.shape, y.shape)
        img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
        preds1 = np.array(preds[0,:,:])
        mask1 = np.array(y[0,:,:])
        print(preds1.max(), preds1.min())
        print(mask1.max(), mask1.min())
        img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
        preds2 = np.array(preds[1,:,:])
        mask2 = np.array(y[1,:,:])
        img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
        preds3 = np.array(preds[2,:,:])
        mask3 = np.array(y[2,:,:])
        ax[0,0].set_title('Image')
        ax[0,1].set_title('Prediction')
        ax[0,2].set_title('Mask')
        ax[1,0].set_title('Image')
        ax[1,1].set_title('Prediction')
        ax[1,2].set_title('Mask')
        ax[2,0].set_title('Image')
        ax[2,1].set_title('Prediction')
        ax[2,2].set_title('Mask')
        ax[0][0].axis("off")
        ax[1][0].axis("off")
        ax[2][0].axis("off")
        ax[0][1].axis("off")
        ax[1][1].axis("off")
        ax[2][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        ax[2][2].axis("off")
        print('*************', img1.shape, preds1.shape, mask1.shape)
        print(mask1.max(), mask1.min())
        print(preds1.max(), preds1.min())
        ax[0][0].imshow(img1)
        ax[0][1].imshow(preds1[0,:,:])
        ax[0][2].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(preds2[0,:,:])
        ax[1][2].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(preds3[0,:,:])
        ax[2][2].imshow(mask3)
        plt.show()
        break