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
        image = Image.open(image).convert('RGB').resize((256, 256))
        mask = Image.open(mask).resize((256, 256))
        
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        sample = {'image':image, 'mask':mask}
        if self.transform:
            transform = transforms.Compose([ 
                transforms.ToTensor(),
            ])
            transform_mask = transforms.Compose([
                transforms.PILToTensor()
            ])
            for sam in sample.keys():
                if sam == 'mask':
                    sample[sam] = transform_mask(sample[sam])
                else:
                    sample[sam] = transform(sample[sam])
                sample[sam] = self.transform(sample[sam])
        else:
            transform = transforms.Compose([ 
                transforms.ToTensor(),
            ])
            transform_mask = transforms.Compose([
                transforms.PILToTensor()
            ])
            for sam in sample.keys():
                if sam == 'mask':
                    sample[sam] = transform_mask(sample[sam])
                else:
                    sample[sam] = transform(sample[sam])
        ### FOR MASKRCNN ###
        # obj_ids = sample['mask'].unique()[1:]
        # num_objs = len(obj_ids)
        # masks = (sample['mask'] == obj_ids[:, None, None]).to(dtype=torch.uint8)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.min(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.min(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # target = {}
        # target['boxes'] = boxes
        # target['labels'] = torch.as_tensor(obj_ids, dtype=torch.int64)
        # target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        return sample['image'], sample['mask']
    

class VastDataLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())
    

    def __getitem__(self, idx):
        pass