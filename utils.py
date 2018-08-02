import multiprocessing

import cv2 as cv
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import train_annot_file, valid_annot_file


def custom_loss(y_true, y_pred):
    return 0.0


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
