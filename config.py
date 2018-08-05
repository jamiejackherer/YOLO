image_h, image_w = 448, 448
num_channels = 3
grid_h, grid_w = 32, 32
grid_size = 32
num_grid = image_h // grid_h
num_box = 1

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']
num_classes = len(labels)
catIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
          63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
catId2idx = dict(zip(catIds, range(num_classes)))

train_image_folder = 'data/train2017'
valid_image_folder = 'data/val2017'
train_annot_file = 'data/annotations/instances_train2017.json'
valid_annot_file = 'data/annotations/instances_val2017.json'

verbose = 1
batch_size = 64
num_epochs = 1000
patience = 50
best_model = 'model.14-2474.5505.hdf5'

lambda_coord = 5.0
lambda_noobj = 0.5

max_boxes = 10  # integer, maximum number of predicted boxes in an image
iou_threshold = 0.5  # real value, "intersection over union" threshold used for NMS filtering
score_threshold = 0.6  # real value, if [ highest class probability score < threshold], then get rid of the corresponding box
