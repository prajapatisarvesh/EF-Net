import os
import pathlib
import sys
import glob


if __name__ == '__main__':
    os.chdir('..')
    try:
        os.chdir('data')
        if len(os.listdir()) != 3:
            raise ValueError('[-] All datasets have not been downloaded')
    except Exception as e:
        print('[-] Data folder has not been populated, please go over the repo\'s readme and populate the data folder.')
        print('\n[+] The program will now exit')
        # sys.exit(1)
    os.chdir('dms-dataset-final')
    ## train images
    train_images = sorted(glob.glob(f'{os.getcwd()}/images/train/*.jpg'))
    train_labels = [a.replace('/images/train', '/labels/train').replace('.jpg', '.png') for a in train_images]
    with open('train.csv', 'w+') as f:
        for i, j in zip(train_images, train_labels):
            f.writelines(f'{i},{j}\n')
    ## test images
    test_images = sorted(glob.glob(f'{os.getcwd()}/images/test/*.jpg'))
    test_labels = [a.replace('/images/test', '/labels/test').replace('.jpg', '.png') for a in test_images]
    with open('test.csv', 'w+') as f:
        for i, j in zip(test_images, test_labels):
            f.writelines(f'{i},{j}\n')
    ## validation images
    validation_images = sorted(glob.glob(f'{os.getcwd()}/images/validation/*.jpg'))
    validation_labels = [a.replace('/images/validation', '/labels/validation').replace('.jpg', '.png') for a in validation_images]
    with open('validation.csv', 'w+') as f:
        for i, j in zip(validation_images, validation_labels):
            f.writelines(f'{i},{j}\n')