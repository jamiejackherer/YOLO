import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence
from pycocotools.coco import COCO

from config import batch_size, image_h, image_w, grid_h, grid_w, num_box, num_classes, num_channels, num_grid
from config import train_image_folder, valid_image_folder, train_annot_file, valid_annot_file, catId2idx


def get_next_box_id(grid_cell):
    for i in range(num_box):
        if grid_cell[i][0] == 0.0:
            return i
    return 0


def get_ground_truth(coco, imgId):
    gt = np.zeros((grid_h, grid_w, num_box, 4 + 1 + num_classes), dtype=np.float32)
    img = coco.loadImgs(ids=[imgId])[0]
    img_height = img['height']
    img_width = img['width']
    annIds = coco.getAnnIds(imgIds=[imgId])
    annos = coco.loadAnns(ids=annIds)
    for anno in annos:
        category_id = anno['category_id']
        xmin, ymin, width, height = anno['bbox']
        xmin = 1.0 * xmin * image_w / img_width
        ymin = 1.0 * ymin * image_h / img_height
        width = 1.0 * width * image_w / img_width
        height = 1.0 * height * image_h / img_height
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        cell_x = int(x_center / grid_w)
        cell_y = int(y_center / grid_h)
        bx = x_center / grid_w - cell_x
        by = y_center / grid_h - cell_y
        bw = width / grid_w
        bh = height / grid_h
        anchor_id = get_next_box_id(gt[cell_y, cell_x])
        gt[cell_y, cell_x, anchor_id, 0] = 1.0
        gt[cell_y, cell_x, anchor_id, 1] = bx
        gt[cell_y, cell_x, anchor_id, 2] = by
        gt[cell_y, cell_x, anchor_id, 3] = bw
        gt[cell_y, cell_x, anchor_id, 4] = bh
        gt[cell_y, cell_x, anchor_id, 5 + catId2idx[category_id]] = 1.0
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
        batch_x = np.empty((length, image_h, image_w, num_channels), dtype=np.float32)
        batch_y = np.empty((length, num_grid, num_grid, num_box, 4 + 1 + num_classes), dtype=np.float32)

        for i_batch in range(length):
            imgId = self.imgIds[i + i_batch]
            img = self.coco.loadImgs(ids=[imgId])[0]
            file_name = img['file_name']
            filename = os.path.join(self.image_folder, file_name)
            image_bgr = cv.imread(filename)
            image_bgr = cv.resize(image_bgr, (image_h, image_w), cv.INTER_CUBIC)

            batch_x[i_batch, :, :] = image_bgr
            batch_y[i_batch, :, :] = get_ground_truth(self.coco, imgId)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.imgIds)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
