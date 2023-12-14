'''
LAST UPDATE: 2023.12.10
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 


'''
import numpy as np
import yaml
import os


# Get the path to the current script
script_path = os.path.abspath(__file__)

# Construct the path to the 'config' folder
config_folder_path = os.path.join(os.path.dirname(script_path), '..', 'config')

# Specify the name of your YAML file
yaml_file_name = 'friction.yaml'

# Construct the full path to the YAML file
yaml_file_path = os.path.join(config_folder_path, yaml_file_name)

with open(yaml_file_path) as file:
    mappings = yaml.safe_load(file)

def convert_to_friction(mask):
        mask = mask.astype(np.float64)
        for i,j in enumerate(mappings.items()):
            # print(i, j[1])
            mask[mask == i] = j[1]
        return mask
        