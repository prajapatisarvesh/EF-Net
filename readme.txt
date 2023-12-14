# EF-Net: End-to-End Friction Estimation Using Intrinsic Imaging and Deep Networks

Submitted by - Sarvesh, Abhinav and Rupesh


This is the code system file for running our end-to-end friction implementation on your system and testing out the results
 
 
## Preparing the dataset folder

```
Download ml-dms dataset from https://github.com/apple/ml-dms-dataset
Download the corresponding images by following https://github.com/JunweiZheng93/MATERobot/
Download VAST dataset from https://github.com/RIVeR-Lab/vast_data/
```

## Checkpoints can be downloaded from -- [https://drive.google.com/drive/folders/1OUrhzb3kesxPONCfVrQcIYcmYhSZ9V2l?usp=drive_link](https://drive.google.com/drive/folders/1OUrhzb3kesxPONCfVrQcIYcmYhSZ9V2l?usp=drive_link)

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

## The deadline was 13th, we submitted it yesterday, since deadline was changed we did major restructuring in the report. If that's allowed we'd like to use 1 time travel day, thanks!

## Thanks to
* https://github.com/JunweiZheng93/MATERobot/ -- for DMS Downloader
* https://github.com/RIVeR-Lab/vast_data/ -- VAST dataset
* https://github.com/apple/ml-dms-dataset -- ml-dms dataset