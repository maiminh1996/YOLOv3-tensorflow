from config import Input_shape, channels, threshold, ignore_thresh
from network_function import YOLOv3
from detect_function import predict
from utils.yolo_utils import read_anchors, read_classes, letterbox_image, resize_image

# from argparse import ArgumentParser
from pathlib import Path
from timeit import time
from timeit import default_timer as timer  # to calculate FPS
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


class YOLO(object):
    def __init__(self):
        self.anchors_path = './model_data/yolo_anchors.txt'
        self.classes_path = './model_data/coco_classes.txt'
        self.class_names = read_classes(self.classes_path)
        self.anchors = read_anchors(self.anchors_path)
        self.threshold = threshold
        self.ignore_thresh = ignore_thresh
        self.INPUT_SIZE = (Input_shape, Input_shape)  # fixed size or (None, None)
        self.is_fixed_size = self.INPUT_SIZE != (None, None)

    def detect_image(self, image):
        start = time.time()

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        if self.is_fixed_size:
            assert self.INPUT_SIZE[0] % 32 == 0, 'Multiples of 32 required'
            assert self.INPUT_SIZE[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image, image_shape = letterbox_image(image, tuple(reversed(self.INPUT_SIZE)))
            # boxed_image, image_shape = resize_image(image, tuple(reversed(self.INPUT_SIZE)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image, image_shape = letterbox_image(image, new_image_size)
            # boxed_image, image_shape = resize_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print("heights, widths:", image_shape)
        image_data /= 255.
        inputs = np.expand_dims(image_data, 0)  # Add batch dimension. #

        # Generate output tensor targets for filtered bounding boxes.
        x = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels])
        # image_shape = np.array([image.size[0], image.size[1]])  # tf.placeholder(tf.float32, shape=[2,])

        # Generate output tensor targets for filtered bounding boxes.
        scale1, scale2, scale3 = YOLOv3(x, len(self.class_names)).feature_extractor()
        scale_total = [scale1, scale2, scale3]

        # detect
        boxes, scores, classes = predict(scale_total, self.anchors, len(self.class_names), image_shape,
                                         score_threshold=self.threshold, iou_threshold=self.ignore_thresh)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
        # tensorboard --logdir="./graphs" --port 6006
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Restore variables from disk.
            epoch = input('Entrer a check point at epoch(%10=0):')
            checkpoint = "/home/minh/stage/saver_model/model" + str(epoch) + ".ckpt"
            try:
                my_abs_path = Path(checkpoint).resolve()
                saver.restore(sess, checkpoint)
            except FileNotFoundError:
                print("Not yet training!")
            else:
                print("already training!")

            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict={x: inputs})

        writer.close()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # Visualisation#################################################################################################
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400  # do day cua BB

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box  # y_min, x_min, y_max, x_max
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))  # (x_min, y_min), (x_max, y_max)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = time.time()
        print(end - start)
        return image


def detect_video(yolo, video_path):
    import cv2
    vid = cv2.VideoCapture(video_path)
    # if not vid.isOpened():
    #    raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()


if __name__ == '__main__':
    detect_img(YOLO())
    # detect_video(YOLO(), 'video.mp4')
