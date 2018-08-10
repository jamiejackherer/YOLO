import multiprocessing
import os

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from config import image_size, valid_annot_file, num_grid, grid_size, train_annot_file, class_weights
from config import lambda_coord, lambda_noobj, lambda_class


def yolo_loss(y_true, y_pred):
    # [None, 13, 13, 1]
    box_conf = K.expand_dims(y_true[..., 0], axis=-1)
    # [None, 13, 13, 2]
    box_xy = y_true[..., 1:3]
    # [None, 13, 13, 2]
    box_wh = y_true[..., 3:5]
    box_wh_half = box_wh / 2.
    box_mins = box_xy - box_wh_half
    box_maxes = box_xy + box_wh_half

    # [None, 13, 13, 1]
    box_conf_hat = K.expand_dims(y_pred[..., 0], axis=-1)
    # [None, 13, 13, 2]
    box_xy_hat = y_pred[..., 1:3]
    # [None, 13, 13, 2]
    box_wh_hat = y_pred[..., 3:5]
    box_wh_half_hat = box_wh_hat / 2.
    box_mins_hat = box_xy_hat - box_wh_half_hat
    box_maxes_hat = box_xy_hat + box_wh_half_hat

    intersect_mins = tf.maximum(box_mins_hat, box_mins)
    intersect_maxes = tf.minimum(box_maxes_hat, box_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # [None, 13, 13]
    true_areas = box_wh[..., 0] * box_wh[..., 1]
    pred_areas = box_wh_hat[..., 0] * box_wh_hat[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    box_conf = iou_scores * box_conf
    # [None, 13, 13, 80]
    box_class = tf.argmax(y_true[..., 5:], -1)
    # [None, 13, 13, 80]
    box_class_hat = y_pred[..., 5:]

    # the position of the ground truth boxes (the predictors)
    # [None, 13, 13, 1]
    coord_mask = K.expand_dims(y_true[..., 0], axis=-1) * lambda_coord
    best_ious = iou_scores
    conf_mask = tf.to_float(best_ious < 0.6) * (1 - y_true[..., 0]) * lambda_noobj
    conf_mask = conf_mask + y_true[..., 0] * lambda_coord
    class_mask = y_true[..., 0] * tf.gather(class_weights, box_class) * lambda_class

    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(box_xy - box_xy_hat) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(box_wh - box_wh_hat) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(box_conf - box_conf_hat) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=box_class, logits=box_class_hat)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 0])
    nb_pred_box = tf.reduce_sum(tf.to_float(box_conf > 0.5) * tf.to_float(box_conf_hat > 0.3))

    """
    Debugging code
    """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)

    return loss


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
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (14, 14, 1)
    boxes -- tensor of shape (14, 14, 4)
    box_class_probs -- tensor of shape (14, 14, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs  # [14, 14, 80]
    print('box_scores.shape: ' + str(box_scores.shape))

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = np.argmax(box_scores, axis=-1)  # [14, 14]
    box_classes = np.expand_dims(box_classes, axis=-1)  # [14, 14, 1]
    print('box_classes.shape: ' + str(box_classes.shape))
    box_class_scores = np.max(box_scores, axis=-1, keepdims=True)  # [14, 14, 1]
    print('box_class_scores.shape: ' + str(box_class_scores.shape))
    print('np.mean(box_class_scores): ' + str(np.mean(box_class_scores)))
    print('np.max(box_class_scores): ' + str(np.max(box_class_scores)))
    print('np.std(box_class_scores): ' + str(np.std(box_class_scores)))

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold  # [14, 14, 1]
    # print('filtering_mask: ' + str(filtering_mask))
    print('filtering_mask.shape: ' + str(filtering_mask.shape))

    # Step 4: Apply the mask to scores, boxes and classes
    scores = box_class_scores[filtering_mask]
    print('scores.shape: ' + str(scores.shape))  # [num_remain]
    boxes = boxes[np.repeat(filtering_mask, 4, axis=2)]  # [num_remain x 4]
    print('boxes.shape: ' + str(boxes.shape))
    classes = box_classes[filtering_mask]  # [num_remain]
    print('classes.shape: ' + str(classes.shape))

    return scores, boxes, classes


# box_xy: [14, 14, 2]
# box_wh: [14, 14, 2]
def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    box_mins = np.clip(box_mins, 0, image_size - 1)
    box_maxes = np.clip(box_maxes, 0, image_size - 1)

    # [14, 14, 4]
    result = np.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ], axis=-1)
    print('result.shape: ' + str(result.shape))
    return result


def scale_box_xy(box_xy):
    result = np.zeros_like(box_xy)
    # shape = 14, 14, 2
    for cell_y in range(num_grid):
        for cell_x in range(num_grid):
            bx = box_xy[cell_y, cell_x, 0]
            by = box_xy[cell_y, cell_x, 1]
            temp_x = (cell_x + bx) * grid_size
            temp_y = (cell_y + by) * grid_size
            result[cell_y, cell_x, 0] = temp_x
            result[cell_y, cell_x, 1] = temp_y
    return result


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), lineType=cv.LINE_AA)
