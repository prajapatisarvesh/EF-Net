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

writer = writer.SummaryWriter(f'runs/unet-segmentation')

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
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')


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
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')



if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_unet_regression()
    # vast_data = VastDataLoader('data/vast_data/labeled_data/vast_data.csv', os.getcwd())
    # vast_data.__getitem__(0)