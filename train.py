'''
LAST UPDATE: 2023.12.10
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


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
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer
import argparse


'''
Train function for UNET Segmentation
Saves weights after every iteration.
'''
def train_unet_segmentation(epochs=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    device = device
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    model = UNet(out_channels=60).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    ### Start epoch
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.type(torch.long)

            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')


'''
Train function for SRCNN Regression
Saves weights after every iteration.
'''
def train_srcnn_regression(epochs=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    device = device
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), mapped=True, transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('checkpoints/epoch_srcnn_99.pt'))
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    # print()
    for epoch in range(100, epochs):
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                predictions = predictions.view(-1, 224, 224)
                loss = loss_fn(predictions, targets.float())
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_srcnn_{epoch}.pt')



'''
Train function for UNET Regression
Saves weights after every iteration.
'''
def train_unet_regression(epochs=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    device = device
    transform = A.Compose([
        A.Resize(224,224),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    data = AppleMLDMSLoader(csv_file='data/dms-dataset-final/train.csv', root_dir=os.getcwd(), mapped=True, transform=transform)
    train_set = int(0.8 * data.__len__())
    test_set = data.__len__() - train_set
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_set, test_set])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    model = UNet(out_channels=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    # print()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train):
            data = data.to(device)
            targets = targets.to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                predictions = predictions.view(-1, 224, 224)
                loss = loss_fn(predictions, targets.float())
            epoch_loss += loss
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_unetr_{epoch}.pt')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

'''
Train End-to-End Friction Estimation
Save weights after every iteration
For good results, train above 50 epochs.
'''
def train_end_to_end_friction_estimation(epochs=500, batch_size=100, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    vast_data = VastDataLoader('data/vast_data/labeled_data/vast_data.csv', os.getcwd())
    train = DataLoader(vast_data, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, shuffle=True)
    model = EndToEndFrictionEstimation().to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(0,epochs):
            epoch_loss = 0.0
            for batch_idx, (data, targets) in enumerate(train):
                data = data.to(device)
                targets = targets.to(device)
                
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)
                epoch_loss += loss
                # backward
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train)} \
                        Loss: {loss}")
            epoch_loss /= len(train)
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            torch.save(model.state_dict(), f'checkpoints/epoch_eee{epoch}.pt')


if __name__ == '__main__':
    '''
    Adding Arg Parser for the program to take in 
    '''
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, dest="model", required=False, default='endtoend', help='Model to train')
    parser.add_argument('--epochs', type=int, dest="epochs", required=False, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, dest="batch_size", required=False, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, dest="lr", required=False, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, dest="device", required=False, default= 'cuda', help='Device to train on')
    parser.add_argument('--log_dir', type=str, dest="log_dir", required=False, default='runs/', help='Log directory for tensorboard')
    
    arg = parser.parse_args()
    writer = writer.SummaryWriter(log_dir='arg.log_dir')
    device = arg.device
    
    if arg.model == 'endtoend':
        train_end_to_end_friction_estimation(epochs=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
        
    elif arg.model == 'unet_reg':
        train_unet_regression(epochs=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    
    elif arg.model == 'unet_seg':
        train_unet_segmentation(epochs=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    
    elif arg.model == 'srcnn':
        train_srcnn_regression(epochs=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    