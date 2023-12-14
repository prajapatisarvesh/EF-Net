# EF-Net: End-to-End Friction Estimation Using Intrinsic Imaging and Deep Networks

Submitted by - Sarvesh, Abhinav and Rupesh

![Model Architecture](https://github.com/prajapatisarvesh/EF-Net/blob/main/misc/EFENET_NEW.jpg)

Note:

`This code is part of CS7180 Final Project, and contains our work for friction estimation and our novel framework for estimating reflectance. This repository contains a lot of different stuff that was used for generation of results, and will be cleaned later on once the coursework is completed and this repository will exclusively serve EF-Net`

This is the code system file for running our end-to-end friction implementation on your system and testing out the results
 
 
## Preparing the dataset folder

```
Download ml-dms dataset from https://github.com/apple/ml-dms-dataset
Download the corresponding images by following https://github.com/JunweiZheng93/MATERobot/
Download VAST dataset from https://github.com/RIVeR-Lab/vast_data/
```

Once the data folder is populated with dataset, execute 
```
cd utils
python3 generate_csv_dms.py
python3 generate_csv_vast.py
``` 

### Testing the scripts

Checkpoints are in `checkpoints` folder and can be used to visualize the results.

`python3 test.py #runs all the models and plots the results`

### Finetuning / Training

`python3 train.py --model [endtoend | unet_reg | unet_seg | srcnn] --epochs num_epochs --batch_size batch_size`

Example

`python3 train.py --model endtoend --epochs 10 --batch_size 100`

*Note due to preprocessing in VAST dataset it may be possible that some datapoints are dropped, it is highly recommended to use a larger batch size (greater than 100).*

### Operating system used - Ubuntu 20.04

### GPU used - Nvidia RTX3090TI and Nvidia RTX4090TI

### Download dataset from vast in dataset -- https://github.com/RIVeR-Lab/vast_terrain_classification

## Thanks to
* https://github.com/JunweiZheng93/MATERobot/ -- for DMS Downloader
* https://github.com/RIVeR-Lab/vast_data/ -- VAST dataset
* https://github.com/apple/ml-dms-dataset -- ml-dms dataset