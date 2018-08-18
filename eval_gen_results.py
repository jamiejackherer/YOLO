# import the necessary packages
import json
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from config import image_size, valid_image_folder, anchors, num_classes, valid_annot_file, catIds
from model import build_model
from utils import ensure_folder, decode_netout, get_best_model

if __name__ == '__main__':
    model = build_model()
    model.load_weights(get_best_model())

    coco = COCO(valid_annot_file)
    imgIds = coco.getImgIds()

    ensure_folder('images')

    results = []
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(ids=[imgId])[0]
        file_name = img['file_name']
        filename = os.path.join(valid_image_folder, file_name)
        image = cv.imread(filename)
        image_bgr = cv.imread(filename)
        orig_h, orig_w = image_bgr.shape[:2]
        image_bgr = cv.resize(image_bgr, (image_size, image_size))
        image_rgb = image_bgr[:, :, ::-1]
        image_rgb = image_rgb / 255.
        image_input = np.expand_dims(image_rgb, 0).astype(np.float32)
        netout = model.predict(image_input)[0]
        boxes = decode_netout(netout, anchors, num_classes)
        for box in boxes:
            x = round(box.xmin * orig_w, 1)
            y = round(box.ymin * orig_h, 1)
            w = round((box.xmax - box.xmin) * orig_w, 1)
            h = round((box.ymax - box.ymin) * orig_h, 1)
            results.append({'image_id': imgId, 'category_id': catIds[box.get_label()], 'bbox': [x, y, w, h],
                            'score': box.get_score()})

        with open('data/eval_results.json', 'w') as file:
            json.dump(results, file)

    K.clear_session()
