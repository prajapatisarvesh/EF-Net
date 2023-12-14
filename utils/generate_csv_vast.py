'''
LAST UPDATE: 2023.12.10
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 
Generates CSV for VAST Dataset

'''
import os
import glob
import sys
import pathlib

if __name__ == '__main__':
    os.chdir('..')
    try:
        os.chdir('data')
        os.chdir('vast_data')
        if len(os.listdir()) != 3:
            raise ValueError('[-] All datasets have not been downloaded')
    except Exception as e:
        print('[-] Data folder has not been populated, please go over the repo\'s readme and populate the data folder.')
        print('\n[+] The program will now exit')
        # sys.exit(1)
    csv_path = os.getcwd()
    os.chdir('labeled_data')
    with open('vast_data.csv', 'w+') as f:
        for dir in sorted(os.listdir()):
            images = sorted(glob.glob(f'{os.getcwd()}/{dir}/*.jpg'))
            spectral = sorted(glob.glob(f'{os.getcwd()}/{dir}/*spec.npy'))
            for i,j in zip(images, spectral):
                f.writelines(f'{i},{j}\n')
        