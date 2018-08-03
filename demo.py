# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf

from config import image_h, image_w, valid_image_folder, max_boxes, iou_threshold, best_model, labels, grid_size
from model import build_model
from utils import ensure_folder, filter_boxes, yolo_boxes_to_corners, scale_boxes, sigmoid, update_box_xy

if __name__ == '__main__':
    model = build_model()
    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    test_path = valid_image_folder
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    num_samples = 1
    samples = random.sample(test_images, num_samples)

    ensure_folder('images')

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        image_shape = image_bgr.shape
        print('image_shape: ' + str(image_shape))
        image_input = cv.resize(image_bgr, (image_h, image_w), cv.INTER_CUBIC)
        image_input = np.expand_dims(image_input, 0).astype(np.float32)
        preds = model.predict(image_input)  # [1, 14, 14, 5, 85]
        # print('preds: ' + str(preds))
        box_confidence = sigmoid(preds[0, :, :, :, 0])
        print('box_confidence: ' + str(box_confidence))
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_xy = sigmoid(preds[0, :, :, :, 1:3])
        box_xy = update_box_xy(box_xy)
        print('box_xy: ' + str(box_xy))
        box_wh = preds[0, :, :, :, 3:5] * grid_size
        print('box_wh: ' + str(box_wh))
        box_class_probs = preds[0, :, :, :, 5:]
        boxes = yolo_boxes_to_corners(box_xy, box_wh)
        print('boxes after to_corners: ' + str(boxes))
        scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs)
        boxes = scale_boxes(boxes, image_shape)
        boxes = np.reshape(boxes, (-1, 4))
        print('boxes after scale: ' + str(boxes))
        scores = np.reshape(scores, (-1))
        classes = np.reshape(classes, (-1))
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
        nms_indices = K.eval(nms_indices)
        print('nms_indices: ' + str(nms_indices))
        scores = scores[nms_indices]
        boxes = boxes[nms_indices]
        # print('classes.shape: ' + str(classes.shape))
        classes = classes[nms_indices]

        for i, cls in enumerate(classes):
            box = boxes[i]
            print(labels[cls])
            y_min, x_min, y_max, x_max = box
            print('y_min={}, x_min={}, y_max={}, x_max={}'.format(y_min, x_min, y_max, x_max))
            cv.rectangle(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0))

        cv.imwrite('images/{}_out.png'.format(i), image_bgr)

    K.clear_session()
