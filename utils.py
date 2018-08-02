import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import train_annot_file, valid_annot_file, lambda_coord


def yolo_loss(y_true, y_pred):
    exists = c = y_true[:, :, :, 0]
    c_hat = y_pred[:, :, :, 0]
    x = y_true[:, :, :, 1]
    x_hat = y_pred[:, :, :, 1]
    y = y_true[:, :, :, 2]
    y_hat = y_pred[:, :, :, 2]
    w = y_true[:, :, :, 3]
    w_hat = y_pred[:, :, :, 3]
    h = y_true[:, :, :, 4]
    h_hat = y_pred[:, :, :, 4]
    cls = y_true[:, :, :, 5:]
    cls_hat = y_pred[:, :, :, 5:]
    loss_xy = K.sum(exists * (K.square(x - x_hat) + K.square(y - y_hat)))
    loss_wh = K.sum(exists * K.square(K.sqrt(w) - K.sqrt(w_hat) + K.square(K.sqrt(h) - K.sqrt(h_hat))))
    loss_conf = K.sum(K.square(c - c_hat))
    loss_class = K.sum(K.square(cls - cls_hat))
    total_loss = lambda_coord * (loss_xy + loss_wh) + loss_conf + loss_class
    return total_loss


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), lineType=cv.LINE_AA)


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def get_example_numbers():
    from pycocotools.coco import COCO
    coco = COCO(train_annot_file)
    num_train_samples = len(coco.getImgIds())
    coco = COCO(valid_annot_file)
    num_valid_samples = len(coco.getImgIds())
    return num_train_samples, num_valid_samples
