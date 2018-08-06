import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import train_annot_file, valid_annot_file, lambda_coord, lambda_noobj, num_grid, num_box, grid_size


def yolo_loss(y_true, y_pred):
    conf = y_true[..., 0]  # [None, 14, 14]
    conf = K.expand_dims(conf, axis=-1)  # [None, 14, 14, 1]
    obj_i_mask = tf.to_float(conf == 1.0)  # [None, 14, 14, 1]
    noobj_i_mask = 1.0 - obj_i_mask  # [None, 14, 14, 1]
    conf_hat = K.expand_dims(y_pred[..., 0], axis=-1)  # [None, 14, 14, 1]
    xy = y_true[..., 1:3]  # [None, 14, 14, 2]
    xy_hat = y_pred[..., 1:3]  # [None, 14, 14, 2]
    wh = y_true[..., 3:5]  # [None, 14, 14, 2]
    wh_hat = y_pred[..., 3:5]  # [None, 14, 14, 2]
    cls = y_true[..., 5:]  # [None, 14, 14, 80]
    cls_hat = y_pred[..., 5:]  # [None, 14, 14, 80]
    loss_xy = lambda_coord * K.sum(obj_i_mask * K.square(xy - xy_hat), axis=(1, 2, 3))  # [None, 14, 14, 2] -> [None]
    loss_wh = lambda_coord * K.sum(obj_i_mask * K.square(K.sqrt(wh) - K.sqrt(wh_hat)),
                                   axis=(1, 2, 3))  # [None, 14, 14, 2] -> [None]
    loss_conf = K.sum(obj_i_mask * K.square(conf - conf_hat), axis=(1, 2, 3))  # [None, 14, 14, 1] -> [None]
    loss_conf += lambda_noobj * K.sum(noobj_i_mask * K.square(conf - conf_hat),
                                      axis=(1, 2, 3))  # [None, 14, 14, 1] -> [None]
    loss_class = K.sum(obj_i_mask * K.square(cls - cls_hat), axis=(1, 2, 3))  # [None, 14, 14, 80] -> [None]
    total_loss = K.mean(loss_xy + loss_wh + loss_conf + loss_class)
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


def filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    print('type(box_confidence): ' + str(type(box_confidence)))
    print(box_confidence)
    print('type(boxes): ' + str(type(boxes)))
    print('type(box_class_probs): ' + str(type(box_class_probs)))
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (14, 14, 5, 1)
    boxes -- tensor of shape (14, 14, 5, 4)
    box_class_probs -- tensor of shape (14, 14, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    print('box_scores.shape: ' + str(box_scores.shape))

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = np.argmax(box_scores, axis=-1)
    box_classes = np.expand_dims(box_classes, axis=-1)
    print('box_classes.shape: ' + str(box_classes.shape))
    box_class_scores = np.max(box_scores, axis=-1, keepdims=True)
    print('box_class_scores.shape: ' + str(box_class_scores.shape))

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold
    print('filtering_mask: ' + str(filtering_mask))
    print('filtering_mask.shape: ' + str(filtering_mask.shape))
    print('type(filtering_mask): ' + str(type(filtering_mask)))

    # Step 4: Apply the mask to scores, boxes and classes
    scores = box_class_scores[filtering_mask]
    print('scores.shape: ' + str(scores.shape))
    print('type(scores): ' + str(type(scores)))
    boxes = boxes[np.repeat(filtering_mask, 4, axis=3)]
    print('boxes.shape: ' + str(boxes.shape))
    print('type(boxes): ' + str(type(boxes)))
    classes = box_classes[filtering_mask]
    print('classes.shape: ' + str(classes.shape))
    print('type(classes): ' + str(type(classes)))

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return np.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = np.stack([height, width, height, width])
    image_dims = np.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def update_box_xy(box_xy):
    result = box_xy.copy()
    # shape = 14, 14, 5, 2
    for cell_y in range(num_grid):
        for cell_x in range(num_grid):
            for j in range(num_box):
                bx = box_xy[cell_y, cell_x, j, 0]
                by = box_xy[cell_y, cell_x, j, 1]
                bx = (cell_x + bx) * grid_size
                by = (cell_y + by) * grid_size
                result[cell_y, cell_x, j, 0] = bx
                result[cell_y, cell_x, j, 1] = by
    return result
