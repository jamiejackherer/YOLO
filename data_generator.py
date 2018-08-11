import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence
from pycocotools.coco import COCO

from config import batch_size, image_size, grid_size, num_classes, num_channels, grid_h, grid_w
from config import train_image_folder, valid_image_folder, train_annot_file, valid_annot_file, catId2idx


def get_ground_truth(coco, imgId):
    gt = np.zeros((num_grid, num_grid, 4 + 1 + num_classes), dtype=np.float32)
    original = coco.loadImgs(ids=[imgId])[0]
    original_height = original['height']
    original_width = original['width']
    annIds = coco.getAnnIds(imgIds=[imgId])
    annos = coco.loadAnns(ids=annIds)
    for anno in annos:
        category_id = anno['category_id']
        xmin, ymin, width, height = anno['bbox']
        xmin = 1.0 * xmin * image_size / original_width
        ymin = 1.0 * ymin * image_size / original_height
        width = 1.0 * width * image_size / original_width
        height = 1.0 * height * image_size / original_height
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        cell_x = int(x_center / grid_size)
        cell_y = int(y_center / grid_size)
        bx = x_center / grid_size - cell_x
        by = y_center / grid_size - cell_y
        bw = width / image_size
        bh = height / image_size
        gt[cell_y, cell_x, 0] = 1.0
        gt[cell_y, cell_x, 1] = bx
        gt[cell_y, cell_x, 2] = by
        gt[cell_y, cell_x, 3] = bw
        gt[cell_y, cell_x, 4] = bh
        gt[cell_y, cell_x, 5 + catId2idx[category_id]] = 1.0
    return gt


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            self.image_folder = train_image_folder
            annot_file = train_annot_file
        else:
            self.image_folder = valid_image_folder
            annot_file = valid_annot_file

        self.coco = COCO(annot_file)
        self.imgIds = self.coco.getImgIds()

        np.random.shuffle(self.imgIds)

    def __len__(self):
        return int(np.ceil(len(self.imgIds) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.imgIds) - i))
        batch_x = np.empty((length, image_size, image_size, num_channels), dtype=np.float32)
        batch_y = np.empty((length, grid_h, grid_w, 4 + 1 + num_classes), dtype=np.float32)

        for i_batch in range(length):
            imgId = self.imgIds[i + i_batch]
            img = self.coco.loadImgs(ids=[imgId])[0]
            file_name = img['file_name']
            filename = os.path.join(self.image_folder, file_name)
            image_bgr = cv.imread(filename)
            image_bgr = cv.resize(image_bgr, (image_size, image_size), cv.INTER_CUBIC)
            image_rgb = image_bgr[:, :, ::-1]

            batch_x[i_batch, :, :] = image_rgb / 255.
            batch_y[i_batch, :, :] = get_ground_truth(self.coco, imgId)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.imgIds)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
