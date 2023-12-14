'''
LAST UPDATE: 2023.12.10
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 

TEST All the models at once

In plotting mode, removing the break will give evaluation metrices for all the images in the test set

'''
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
from utils.convert_to_friction import convert_to_friction
from torchmetrics import Dice
from torchmetrics.regression import MeanSquaredError


'''
Collate function for clearning the vast dataset
'''
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

'''
Tests all the proposed methods
1. EF-NET
2. SRCNN
3. UNETR
4. UNETS
'''
def test_all_methods():
    # Define device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Define model and dataloader
    model_efnet = EndToEndFrictionEstimation().to(device)
    
    vast_data = VastDataLoader('data/vast_data/labeled_data/vast_data.csv', os.getcwd())
    train_set = int(0.7 * vast_data.__len__())
    test_set = vast_data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(vast_data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=500, pin_memory=True, shuffle=True)
    model_efnet.load_state_dict(torch.load('checkpoints/EFNET.pt',map_location='cuda:0'))
    # Test a small batch of images for EF-Net
    for batch_idx, (data, targets) in enumerate(test):
        x = data.to(device)
        x = torch.transpose(x, 2, 3)
        y = targets.to('cpu')
        print(x.shape, y.shape)
        preds = model_efnet(x).to('cpu').detach()
        img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
        print(img1.shape)
        preds1 = np.array(preds[0,:])
        mask1 = np.array(y[0,:])
        print(preds1.max(), preds1.min())
        print(mask1.max(), mask1.min())
        img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
        preds2 = np.array(preds[1,:])
        mask2 = np.array(y[1,:])
        img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
        preds3 = np.array(preds[2,:])
        mask3 = np.array(y[2,:])
        ### Plot function
        fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
        ax[0,0].set_title('Image')
        ax[0,1].set_title('Prediction')
        ax[0,2].set_title('True')
        ax[1,0].set_title('Image')
        ax[1,1].set_title('Prediction')
        ax[1,2].set_title('True')
        ax[2,0].set_title('Image')
        ax[2,1].set_title('Prediction')
        ax[2,2].set_title('True')
        ax[0][0].axis("off")
        ax[1][0].axis("off")
        ax[2][0].axis("off")
        ax[0][1].axis("off")
        ax[1][1].axis("off")
        ax[2][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        ax[2][2].axis("off")
        ax[0][0].imshow(img1)
        ax[0][1].plot(preds1)
        ax[0][2].plot(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].plot(preds2)
        ax[1][2].plot(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].plot(preds3)
        ax[2][2].plot(mask3)
        plt.show()
        break
    
    ### Non spectral stuff and plots 
    model_srcnn = SRCNN().to(device)
    model_unetr = UNet(out_channels=1).to(device)
    model_unets = UNet(out_channels=60).to(device)
    model_end2end = EndToEndFrictionEstimation().to(device)
    model_srcnn.load_state_dict(torch.load('checkpoints/SRCNN.pt'))
    model_unetr.load_state_dict(torch.load('checkpoints/UNET_REGRESSION.pt'))
    model_unets.load_state_dict(torch.load('checkpoints/UNET_SEGMENTATION.pt'))
    # model_end2end.load_state_dict(torch.load('checkpoints/epoch_eee88.pt'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data_seg = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), mapped=False, transform=transform)
    train_set_seg = int(0.8 * data_seg.__len__())
    test_set_seg = data_seg.__len__() - train_set_seg
    train_dataset, test_dataset = torch.utils.data.random_split(data_seg, [train_set_seg, test_set_seg])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=True)
    dice_unetseg = 0.0
    dice_unetreg = 0.0
    dice_srcnn = 0.0
    counter = 0
    for x,y in test:
        x = x.to(device)
        fig , ax =  plt.subplots(3, 6, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds_1 = model_srcnn(x).to('cpu').detach()
        preds_2 = model_unetr(x).to('cpu').detach()
        preds = torch.argmax(softmax(model_unets(x)), axis=1).to('cpu').detach()
        z = torch.tensor(convert_to_friction(y.numpy()))
        metric = MeanSquaredError()
        # print(type(z), type(preds), type(preds_1))
        dice_unetseg += metric(preds, y)
        dice_unetreg += metric(preds_2.view(-1,224,224), z)
        dice_srcnn += metric(preds_1.view(-1,224,224), z)
        counter += 1
        img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
        preds1_ur = np.array(preds[0,:,:])
        preds1_s = np.array(preds_1[0,:,:]) 
        preds1_u = np.array(preds_2[0,:,:])
        mask1 = np.array(y[0,:,:])
        mask1_ = np.array(z[0,:,:])
        print(mask1.max(), mask1.min())
        img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
        # preds2 = np.array(preds[1,:,:])
        preds2_ur = np.array(preds[1,:,:])
        preds2_s = np.array(preds_1[1,:,:])
        preds2_u = np.array(preds_2[1,:,:])
        mask2 = np.array(y[1,:,:])
        mask2_ = np.array(z[1,:,:])
        img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
        # preds3 = np.array(preds[2,:,:])
        preds3_ur = np.array(preds[2,:,:])
        preds3_s = np.array(preds_1[2,:,:])
        preds3_u = np.array(preds_2[2,:,:])
        mask3 = np.array(y[2,:,:])
        mask3_ = np.array(z[2,:,:])
        
        ax[0,0].set_title('Image')
        ax[0,1].set_title('Prediction UNETR')
        ax[0,2].set_title('Prediction SRCNN')
        ax[0,3].set_title('Mask Regression')
        ax[0,4].set_title('Prediction UNETS')
        ax[0,5].set_title('Mask Segmentation')
        ax[1,0].set_title('Image')
        ax[1,1].set_title('Prediction UNETR')
        ax[1,2].set_title('Prediction SRCNN')
        ax[1,3].set_title('Mask Regression')
        ax[1,4].set_title('Prediction UNETS')
        ax[1,5].set_title('Mask Segmentation')
        ax[2,0].set_title('Image')
        ax[2,1].set_title('Prediction UNETR')
        ax[2,2].set_title('Prediction SRCNN')
        ax[2,3].set_title('Mask Regression')
        ax[2,4].set_title('Prediction UNETS')
        ax[2,5].set_title('Mask Segmentation')
        ax[0][0].axis("off")
        ax[1][0].axis("off")
        ax[2][0].axis("off")
        ax[0][1].axis("off")
        ax[1][1].axis("off")
        ax[2][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        ax[2][2].axis("off")
        ax[0][3].axis("off")
        ax[1][3].axis("off")
        ax[2][3].axis("off")
        ax[0][4].axis("off")
        ax[1][4].axis("off")
        ax[2][4].axis("off")
        ax[0][5].axis("off")
        ax[1][5].axis("off")
        ax[2][5].axis("off")
        # print('*************', img1.shape, preds1.shape, mask1.shape)
        print(mask1.max(), mask1.min())
        # print(preds1.max(), preds1.min())
        ax[0][0].imshow(img1)
        ax[0][1].imshow(preds1_u[0,:,:])
        ax[0][2].imshow(preds1_s[0,:,:])
        ax[0][3].imshow(mask1_)
        ax[0][4].imshow(preds1_ur)
        ax[0][5].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(preds2_u[0,:,:])
        ax[1][2].imshow(preds2_s[0,:,:])
        ax[1][3].imshow(mask2_)
        ax[1][4].imshow(preds2_ur)
        ax[1][5].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(preds3_u[0,:,:])
        ax[2][2].imshow(preds3_s[0,:,:])
        ax[2][3].imshow(mask3_)
        ax[2][4].imshow(preds3_ur)
        ax[2][5].imshow(mask3)
        plt.show()
        break
    # # model = SRCNN().to(device)
    # model = UNet(out_channels=1).to(device)
    # model.load_state_dict(torch.load('checkpoints/epoch_unetr_112.pt'))
    # # model.load_state_dict(torch.load('checkpoints/epoch_srcnn_99.pt'))
    # plot_reg(test, model, device)
    # dice_unetreg /= counter
    # dice_unetseg /= counter
    # dice_srcnn /= counter
    # print('Dice UNETR Regression:', dice_unetreg)
    # print('Dice UNET Segmentation:', dice_unetseg)
    # print('Dice SRCNN:', dice_srcnn)


if __name__ == '__main__':
    test_all_methods()