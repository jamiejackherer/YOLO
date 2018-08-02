import json
import os
import zipfile

import cv2 as cv
from tqdm import tqdm


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    extract('train2017')
    extract('val2017')
    extract('annotations_trainval2017')

