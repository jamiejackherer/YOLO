# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import image_size, valid_image_folder, best_model, labels, grid_size, score_threshold
from model import build_model
from utils import ensure_folder, filter_boxes, yolo_boxes_to_corners, scale_box_xy, draw_str

if __name__ == '__main__':
    model = build_model()
    model_weights_path = os.path.join('models', best_model)
    model.load_weights(model_weights_path)

    test_path = valid_image_folder
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    num_samples = 1
    random.seed(1)
    samples = random.sample(test_images, num_samples)

    ensure_folder('images')

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        image_shape = image_bgr.shape
        print('image_shape: ' + str(image_shape))
        image_input = cv.resize(image_bgr, (image_size, image_size), cv.INTER_CUBIC)
        image_input = np.expand_dims(image_input, 0).astype(np.float32)
        preds = model.predict(image_input)  # [1, 14, 14, 85]
        box_confidence = preds[0, :, :, 0]
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_confidence = np.clip(box_confidence, 0.0, 1.0)
        print('np.mean(box_confidence): ' + str(np.mean(box_confidence)))
        print('np.max(box_confidence): ' + str(np.max(box_confidence)))
        print('np.std(box_confidence): ' + str(np.std(box_confidence)))
        box_xy = preds[0, :, :, 1:3]
        box_xy = np.clip(box_xy, 0.0, 1.0)
        box_xy = scale_box_xy(box_xy)
        print('np.mean(box_xy): ' + str(np.mean(box_xy)))
        print('np.std(box_xy): ' + str(np.std(box_xy)))
        box_wh = preds[0, :, :, 3:5]
        box_wh = np.clip(box_wh, 0.0, 1.0)
        box_wh = box_wh * image_size
        print('np.mean(box_wh): ' + str(np.mean(box_wh)))
        print('np.max(box_wh): ' + str(np.max(box_wh)))
        print('np.std(box_wh): ' + str(np.std(box_wh)))
        box_class_probs = preds[0, :, :, 5:]
        boxes = yolo_boxes_to_corners(box_xy, box_wh)
        print('boxes: ' + str(boxes))
        print('boxes.shape: ' + str(boxes.shape))
        scores, boxes, classes = filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
        boxes = np.reshape(boxes, (-1, 4))
        print('boxes after reshape: ' + str(boxes))

        for j, cls in enumerate(classes):
            box = boxes[j]
            label = labels[cls]
            print(label)
            x_min, y_min, x_max, y_max = box
            print('x_min={}, y_min={}, x_max={}, y_max={}'.format(x_min, y_min, x_max, y_max))
            cv.rectangle(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0))
            draw_str(image_bgr, (int(x_min), int(y_min)), label)

        cv.imwrite('images/{}_out.png'.format(i), image_bgr)

    K.clear_session()
