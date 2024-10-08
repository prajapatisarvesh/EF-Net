# Predictive Mapping of Spectral Signatures from RGB Imagery for Off-Road Terrain Analysis

![Model Architecture](https://github.com/prajapatisarvesh/EF-Net/blob/main/misc/EFENET_NEW.jpg)

This is the code system file for running our end-to-end friction implementation on your system and testing out the results
 
 
## Preparing the dataset folder

```
Download ml-dms dataset from https://github.com/apple/ml-dms-dataset
Download the corresponding images by following https://github.com/JunweiZheng93/MATERobot/
Download VAST dataset from https://github.com/RIVeR-Lab/vast_data/
```

Checkpoints can be downloaded from -- [https://drive.google.com/drive/folders/1OUrhzb3kesxPONCfVrQcIYcmYhSZ9V2l?usp=drive_link](https://drive.google.com/drive/folders/1OUrhzb3kesxPONCfVrQcIYcmYhSZ9V2l?usp=drive_link)

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

## Citation

<hr/>

If you find this code useful, please consider citing our paper:

```
@misc{prajapati2024predictivemappingspectralsignatures,
      title={Predictive Mapping of Spectral Signatures from RGB Imagery for Off-Road Terrain Analysis}, 
      author={Sarvesh Prajapati and Ananya Trivedi and Bruce Maxwell and Taskin Padir},
      year={2024},
      eprint={2405.04979},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2405.04979}, 
}
```

<hr/>

## Thanks to
* https://github.com/JunweiZheng93/MATERobot/ -- for DMS Downloader
* https://github.com/RIVeR-Lab/vast_data/ -- VAST dataset
* https://github.com/apple/ml-dms-dataset -- ml-dms dataset
