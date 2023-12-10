import torch
import os
import torch.nn as nn
from PIL import Image
from base.base_data_loader import BaseDataLoader
import torchvision.transforms as transforms 
import  matplotlib.pyplot as plt
import numpy as np

class AppleMLDMSLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())

    def __getitem__(self, idx):
        image = self.csv_dataframe.iloc[idx, 0]
        mask = self.csv_dataframe.iloc[idx, 1]
        image = np.array(Image.open(image).convert('RGB'))
        mask = np.array(Image.open(mask))
        
        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        return image, mask
    

class VastDataLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())
    

    def __getitem__(self, idx):
        image = self.csv_dataframe.iloc[idx, 0]
        spectral_data = self.csv_dataframe.iloc[idx, 1]
        image = Image.open(image)
        spectral_data = np.load(spectral_data)
        print(spectral_data.shape)