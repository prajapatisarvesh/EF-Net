import torch
import os
import torch.nn as nn
from PIL import Image
from base.base_data_loader import BaseDataLoader
import torchvision.transforms as transforms 
import  matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_to_friction import convert_to_friction


class AppleMLDMSLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, mapped=False, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        self.mapped = mapped
        print("[+] Data Loaded with rows: ", super().__len__())
        
        
    def __getitem__(self, idx):
        image = self.csv_dataframe.iloc[idx, 0]
        mask = self.csv_dataframe.iloc[idx, 1]
        image = np.array(Image.open(image).convert('RGB'))
        mask = np.array(Image.open(mask))
        if self.mapped:
            mask = convert_to_friction(mask)
        
        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        return image, mask
    

class VastDataLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=None)
        self.transform_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.min_ind = 250
        self.max_ind = 1800
        self.dark_min = torch.tensor(np.load(self.root_dir+'/data/vast_data/calibration/dark_min.npy'))
        self.light_max = torch.tensor(np.load(self.root_dir+'/data/vast_data/calibration/light_max.npy'))
        self.light_max = torch.clamp(self.light_max, min=2000)
        self.dark_min = self.dark_min[self.min_ind:self.max_ind]
        self.light_max = self.light_max[self.min_ind:self.max_ind]
        # print(self.dark_min.shape, self.light_max.shape)
        print("[+] Data Loaded with rows: ", super().__len__())
    

    def __getitem__(self, idx):
        image = self.csv_dataframe.iloc[idx, 0]
        spectral_data = self.csv_dataframe.iloc[idx, 1]
        image = Image.open(image)
        spectral_data = np.load(spectral_data)[1::2]
        transformed_image = self.transform_(image)
        if np.mean(spectral_data) <= 1800:
            return None
        spectral_data = torch.tensor(spectral_data)
        spectral_data = spectral_data[self.min_ind:self.max_ind]
        data = torch.clamp((spectral_data - self.dark_min) / (self.light_max - self.dark_min + 1e-6), min=0, max=1)
        return transformed_image, data.float()