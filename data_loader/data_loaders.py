import torch
import os
import torch.nn as nn
from PIL import Image
from base.base_data_loader import BaseDataLoader

class AppleMLDMSLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform)
        print("[+] Data Loaded with rows: ", super().__len__())

    def __getitem__():
        pass