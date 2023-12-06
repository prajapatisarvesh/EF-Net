import torch
import os
import torch.nn as nn
from PIL import Image
from base.base_data_loader import BaseDataLoader
import torchvision.transforms as transforms 
import  matplotlib.pyplot as plt

class AppleMLDMSLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())

    def __getitem__(self, idx):
        image = self.csv_dataframe.iloc[idx, 0]
        label = self.csv_dataframe.iloc[idx, 1]
        image = Image.open(image).convert('RGB')
        label = Image.open(label)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        sample = {'image':image, 'label':label}
        if self.transform:
            transform = transforms.Compose([ 
                transforms.PILToTensor() 
            ])
            for sam in sample.keys():
                sample[sam] = transform(sample[sam])
                sample[sam] = self.transform(sample[sam])
        else:
            transform = transforms.Compose([ 
                transforms.PILToTensor() 
            ])
            for sam in sample.keys():
                sample[sam] = transform(sample[sam])
        return sample