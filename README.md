## YOLO

YOLOv2 [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) 的 Keras 实现。

## 依赖项

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

MSCOCO 2017 数据集，请按照[说明](http://cocodataset.org/#download) 下载 train2017.zip, val2017.zip, annotations_trainval2017.zip 放入 data 目录。

```bash
$ wget http://images.cocodataset.org/zips/train2017.zip && wget http://images.cocodataset.org/zips/val2017.zip && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

![image](https://github.com/foamliu/YOLO/raw/master/images/COCO_2017.png)

## 用法

### 数据预处理
提取123,287个训练图像，并将它们分开（53,879个用于训练，7,120个用于验证）：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想在培训期间进行可视化，请在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

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

### 数据增强

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

