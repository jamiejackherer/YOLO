from pycocotools.coco import COCO
import pylab
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from config import train_annot_file

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

if __name__ == '__main__':
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running demo for *%s* results.' % (annType))

    # initialize COCO ground truth api
    cocoGt = COCO(train_annot_file)

    # initialize COCO detections api
    cocoDt = cocoGt.loadRes('data/eval_results.json')

    imgIds = sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
