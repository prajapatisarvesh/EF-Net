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
    for dir in sorted(os.listdir()):
        pass