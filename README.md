## YOLO

YOLOv2 [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) of Keras achieve。

## Dependency

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## data set

MSCOCO 2017 For the data set, please follow the [Description] (http://cocodataset.org/#download) download train2017.zip, val2017.zip, annotations_trainval2017.zip into the data directory.

```bash
$ wget http://images.cocodataset.org/zips/train2017.zip && wget http://images.cocodataset.org/zips/val2017.zip && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

![image](https://github.com/foamliu/YOLO/raw/master/images/COCO_2017.png)

## usage

### Data preprocessing
Extract 123,287 training images and separate them (118, 287 for training and 5,000 for verification):
```bash
$ python pre-process.py
```

### Training
```bash
$ python train.py
```

If you want to visualize during training, please run in the terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Performance Evaluation

```bash
$ python eval_gen_results.py
$ python eval.py
```

mAP = 30.3

KPI|value|
|---|---|
 Average Precision  (AP)| @[ IoU=0.50:0.95 + area=   all + maxDets=100 ] = 0.120|
 Average Precision  (AP)| @[ IoU=0.50      + area=   all + maxDets=100 ] = 0.189|
 Average Precision  (AP)| @[ IoU=0.75      + area=   all + maxDets=100 ] = 0.131|
 Average Precision  (AP)| @[ IoU=0.50:0.95 + area= small + maxDets=100 ] = 0.000|
 Average Precision  (AP)| @[ IoU=0.50:0.95 + area=medium + maxDets=100 ] = 0.046|
 Average Precision  (AP)| @[ IoU=0.50:0.95 + area= large + maxDets=100 ] = 0.303|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area=   all + maxDets=  1 ] = 0.115|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area=   all + maxDets= 10 ] = 0.134|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area=   all + maxDets=100 ] = 0.134|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area= small + maxDets=100 ] = 0.000|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area=medium + maxDets=100 ] = 0.044|
 Average Recall     (AR)| @[ IoU=0.50:0.95 + area= large + maxDets=100 ] = 0.357|

### Demo
Download [pre-trained model] (https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.11-0.6262.hdf5) in the models directory and execute:
```bash
$ python demo.py
```

|1|2|3|4|
|---|---|---|---|
|![image](https://github.com/foamliu/YOLO/raw/master/images/0_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/5_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/10_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/15_out.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/1_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/6_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/11_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/16_out.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/2_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/7_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/12_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/17_out.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/3_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/8_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/13_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/18_out.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/4_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/9_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/14_out.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/19_out.png)|

### Data enhancement

```bash
$ python augmentor.py
```
|before|after|
|---|---|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_0.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_0.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_1.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_1.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_2.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_2.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_3.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_3.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_4.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_4.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_5.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_5.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_6.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_6.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_7.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_7.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_8.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_8.png)|
|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_before_9.png)|![image](https://github.com/foamliu/YOLO/raw/master/images/imgaug_after_9.png)|

