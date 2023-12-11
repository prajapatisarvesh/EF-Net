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
from model.model import UNet, SRCNN
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer

writer = writer.SummaryWriter(log_dir='runs/unet_reg')

def train_unet_segmentation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=True)
    model = UNet(out_channels=60).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    NUM_EPOCHS  = 10

    # print()
    for epoch in range(NUM_EPOCHS):
        # print(epoch)
        # loop = tqdm(enumerate(train),total=len(train))
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                
                predictions = predictions.view(-1, 224, 224)
                # print(data.shape, targets.shape, predictions.shape)
                loss = loss_fn(predictions, targets.float())
                # print(loss)
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
            # update tqdm loop
            # print(loss)
            # loop.set_postfix(loss=loss.item())
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')

def train_srcnn_regression():
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
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('checkpoints/epoch_srcnn_6.pt'))
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    NUM_EPOCHS  = 100

    # print()
    for epoch in range(NUM_EPOCHS):
        # print(epoch)
        # loop = tqdm(enumerate(train),total=len(train))
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                
                predictions = predictions.view(-1, 224, 224)
                # print(data.shape, targets.shape, predictions.shape)
                loss = loss_fn(predictions, targets.float())
                # print(loss)
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
            # update tqdm loop
            # print(loss)
            # loop.set_postfix(loss=loss.item())
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_srcnn_{epoch}.pt')


def test_srcnn_regression():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    device='cpu'
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), mapped=True, transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=True)
    model = SRCNN().to(device)
    # model = UNet(out_channels=1).to(device)
    # model.load_state_dict(torch.load('checkpoints/epoch_unetr_71.pt'))
    model.load_state_dict(torch.load('checkpoints/epoch_srcnn_74.pt'))
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

def train_unet_regression():
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
    model = UNet(out_channels=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    NUM_EPOCHS  = 500

    # print()
    for epoch in range(NUM_EPOCHS):
        # print(epoch)
        # loop = tqdm(enumerate(train),total=len(train))
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                
                predictions = predictions.view(-1, 224, 224)
                # print(data.shape, targets.shape, predictions.shape)
                loss = loss_fn(predictions, targets.float())
                # print(loss)
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
            # update tqdm loop
            # print(loss)
            # loop.set_postfix(loss=loss.item())
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_unetr_{epoch}.pt')


def train_end_to_end_friction_estimation():
    

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train_unet_regression()
    # train_srcnn_regression()
    test_srcnn_regression()
    # vast_data = VastDataLoader('data/vast_data/labeled_data/vast_data.csv', os.getcwd())
    # vast_data.__getitem__(0)