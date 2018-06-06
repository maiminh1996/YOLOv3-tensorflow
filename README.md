# YOLOv3 tensorflow 
> Build a real-time bounding-box object detection system for the boat (using fine-tuning in tensorflow based on YOLOv3-416 weights trained en COCO dataset). Then use my own data set for distinguish different type of boat 


Inspired by https://github.com/allanzelener/YAD2K and https://github.com/qqwweee/keras-yolo3


##  Architecture


![Imgur](https://i.imgur.com/ToEbljZ.png)


| Input to CCNs(Features block) | General | 3 Scales | Features |
|-------------------------------|---------|----------|--------- |
| ![Imgur](https://i.imgur.com/BVWAq2e.png) | ![Imgur](https://i.imgur.com/7MKumGI.png) | ![Imgur](https://i.imgur.com/WfaG4Cw.png) | ![Imgur](https://i.imgur.com/C6DjsB9.jpg) |


## Test
1. Clone this folder
2. Transfomer the pre-trained weights in Darknet to keras (may be skip this etape to etape 3)
  <ul>
  <li>wget https://pjreddie.com/media/files/yolov3.weights </li>
  <li>python3 convert.py yolov3.cfg yolov3.weights yolov3.h5</li>
  <li>python3 yolo.py </li>
  </ul>
  
  
3. Or download the pre-trained weights in keras from https://drive.google.com/open?id=1cVWJE1hv1M_KxzyJN6NE52L2JKqjW133
4. Run python3 propagation.py 


**Results** (La Rochelle, la belle ville :) )


| YOLOv3-608 | YOLOv3-416 | YOLOv3-320 |
|------------|------------|------------|
| ![608](https://i.imgur.com/d6wCvfx.jpg) | ![416](https://i.imgur.com/jL2gnXW.jpg) | ![320](https://i.imgur.com/XlOdq1N.jpg) |


## Train for your own dataset


1. Run python3 boat_annotation.py to get 3 files: bateau_train.txt, bateau_valid.txt, bateau_test.txt
  <ul>
  <li>In each file contains path_to_image obj1 obj2 ...</li>
  <li>With obj1: x1_min, y1_min, x1_max, y1_max</li>
  </ul>
  
  
2. Run python3 train.py
3. In propagation.py, modify classes_path to boat_classes.txt
4. Run python3 propagation.py
5. Enjoy your results!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


