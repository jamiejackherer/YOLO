# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf

from config import image_h, image_w, valid_image_folder, max_boxes, iou_threshold, best_model
from model import build_model
from utils import ensure_folder, filter_boxes, yolo_boxes_to_corners, scale_boxes

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
        image_input = cv.resize(image_bgr, (image_h, image_w), cv.INTER_CUBIC)
        image_input = np.expand_dims(image_input, 0).astype(np.float32)
        preds = model.predict(image_input)  # [1, 14, 14, 5, 85]
        box_confidence = preds[0, :, :, :, 0]
        box_xy = preds[0, :, :, :, 1:3]
        box_wh = preds[0, :, :, :, 3:5]
        box_class_probs = preds[0, :, :, :, 5:]
        boxes = yolo_boxes_to_corners(box_xy, box_wh)
        scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs)
        boxes = scale_boxes(boxes, image_shape)
        nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
        scores = scores[nms_indices]
        boxes = boxes[nms_indices]
        classes = classes[nms_indices]

        for box in boxes:
            y_min, x_min, y_max, x_max = box
            cv.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0))

        cv.imwrite('images/{}_out.png'.format(i), image_bgr)

    K.clear_session()
