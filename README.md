# YOLOv3 tensorflow 
> Build a real-time bounding-box object detection system for the boat (using fine-tuning in tensorflow based on YOLOv3-416 weights trained en COCO dataset). Then use my own data set for distinguish different type of boat 


## Test
1. Clone this folder
2. Download the weights in keras from https://drive.google.com/open?id=1cVWJE1hv1M_KxzyJN6NE52L2JKqjW133
3. Run python3 propagation.py 


**Results** (La Rochelle, la belle ville :) )


| YOLOv3-608 | YOLOv3-416 | YOLOv3-320 |
|------------|------------|------------|
| ![608](https://i.imgur.com/d6wCvfx.jpg) | ![416](https://i.imgur.com/jL2gnXW.jpg) | ![320](https://i.imgur.com/XlOdq1N.jpg) |


## Train


1. Run python3 boat_annotation.py to get 3 files: bateau_train.txt, bateau_valid.txt, bateau_test.txt
 *In each file contains path_to_image obj1 obj2 ...
 *With obj1: x1_min, y1_min, x1_max, y1_max
2. Run python3 train.py
3. Run python3 test.py(propagation) A venir!


