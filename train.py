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



def train_unet_segmentation(NUM_EPOCHS=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    NUM_EPOCHS  = NUM_EPOCHS


    for epoch in range(NUM_EPOCHS):
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
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')

def train_srcnn_regression(NUM_EPOCHS=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    NUM_EPOCHS  = NUM_EPOCHS

    # print()
    for epoch in range(100, NUM_EPOCHS):
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
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
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

def train_unet_regression(NUM_EPOCHS=500, batch_size=100, lr=3e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    NUM_EPOCHS  = NUM_EPOCHS

    # print()
    for epoch in range(NUM_EPOCHS):
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
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                      Loss: {loss}")
        epoch_loss /= len(train)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        torch.save(model.state_dict(), f'checkpoints/epoch_unetr_{epoch}.pt')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train_end_to_end_friction_estimation(NUM_EPOCHS=500, batch_size=100, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    vast_data = VastDataLoader('data/vast_data/labeled_data/vast_data.csv', os.getcwd())
    train = DataLoader(vast_data, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True, shuffle=True)
    model = EndToEndFrictionEstimation().to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(0,NUM_EPOCHS):
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
                    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train)} \
                        Loss: {loss}")
            epoch_loss /= len(train)
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            torch.save(model.state_dict(), f'checkpoints/epoch_eee{epoch}.pt')


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('--model', type=str, dest="model", required=True, default='endtoend', help='Model to train')
    parser.add_argument('--epochs', type=int, dest="epochs", required=False, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, dest="batch_size", required=False, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, dest="lr", required=False, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, dest="device", required=False, default= 'cuda', help='Device to train on')
    parser.add_argument('--log_dir', type=str, dest="log_dir", required=False, default= 'runs/unet_reg', help='Log directory for tensorboard')
    
    arg = parser.parse_args()
    writer = writer.SummaryWriter(log_dir='arg.log_dir')
    device = arg.device
    
    if arg.model == 'endtoend':
        train_end_to_end_friction_estimation(epoch=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
        
    elif arg.model == 'unet_reg':
        train_unet_regression(epoch=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    
    elif arg.model == 'unet_seg':
        train_unet_segmentation(epoch=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    
    elif arg.model == 'srcnn':
        train_srcnn_regression(epoch=arg.epochs, batch_size=arg.batch_size, lr=arg.lr, device=arg.device)
    