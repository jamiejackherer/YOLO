import os

import cv2 as cv
import numpy as np
from keras.utils import Sequence
from pycocotools.coco import COCO

from augmentor import aug_image
from config import batch_size, image_h, image_w, grid_h, grid_w, num_classes, num_channels, num_box, grid_size, \
    train_image_folder, valid_image_folder, train_annot_file, valid_annot_file, catId2idx, anchors
from utils import BoundBox, bbox_iou

anchor_boxes = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(int(len(anchors) // 2))]


def get_ground_truth(coco, imgId):
    gt = np.zeros((grid_h, grid_w, num_box, 4 + 1 + num_classes), dtype=np.float32)
    annIds = coco.getAnnIds(imgIds=[imgId])
    annos = coco.loadAnns(ids=annIds)
    for anno in annos:
        category_id = anno['category_id']
        xmin, ymin, width, height = anno['bbox']
        xmin = 1.0 * xmin * image_w
        ymin = 1.0 * ymin * image_h
        width = 1.0 * width * image_w
        height = 1.0 * height * image_h
        center_x = xmin + width / 2.
        center_x = center_x / grid_size
        center_y = ymin + height / 2.
        center_y = center_y / grid_size
        cell_x = int(np.floor(center_x))
        cell_y = int(np.floor(center_y))
        center_w = width / grid_size
        center_h = height / grid_size
        box = [center_x, center_y, center_w, center_h]

        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou = -1

        shifted_box = BoundBox(0,
                               0,
                               center_w,
                               center_h)

        for i in range(len(anchor_boxes)):
            anchor = anchor_boxes[i]
            iou = bbox_iou(shifted_box, anchor)

            if max_iou < iou:
                best_anchor = i
                max_iou = iou

        # assign ground truth x, y, w, h, confidence and class probs
        gt[cell_y, cell_x, best_anchor, 0] = 1.0
        gt[cell_y, cell_x, best_anchor, 1:5] = box
        gt[cell_y, cell_x, best_anchor, 5 + catId2idx[category_id]] = 1.0
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
        self.num_samples = len(self.imgIds)

        np.random.shuffle(self.imgIds)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (self.num_samples - i))
        batch_x = np.empty((length, image_h, image_w, num_channels), dtype=np.float32)
        batch_y = np.empty((length, grid_h, grid_w, num_box, 4 + 1 + num_classes), dtype=np.float32)

        for i_batch in range(length):
            imgId = self.imgIds[i + i_batch]
            img = self.coco.loadImgs(ids=[imgId])[0]
            file_name = img['file_name']
            filename = os.path.join(self.image_folder, file_name)
            image_bgr = cv.imread(filename)
            image_bgr = cv.resize(image_bgr, (image_h, image_w))
            image_rgb = image_bgr[:, :, ::-1]
            if self.usage == 'train':
                image, boxes = aug_image(image, boxes, jitter=True)
            else:
                image, boxes = aug_image(image, boxes, jitter=False)

            batch_x[i_batch, :, :] = image_rgb / 255.
            batch_y[i_batch, :, :] = get_ground_truth(self.coco, imgId)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.imgIds)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
