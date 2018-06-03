import colorsys
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def get_training_data(annotation_path, data_path, input_shape, max_boxes=100, load_previous=True):
    """
    processes the data into standard shape
    :param annotation_path: path_to_image box1,box2,...,boxN with boxX: x_min,y_min,x_max,y_max,class_index
    :param data_path: saver at "/home/minh/stage/train.npz"
    :param input_shape: (416, 416)
    :param max_boxes: 100: maximum number objects of an image
    :param load_previous: for 2nd, 3th, .. using
    :return: image_data [N, 416, 416, 3] not yet normalized, N: number of image
             box_data: box format: [N, 100, 6], 100: maximum number of an image
                                                6: top_left{x_min,y_min},bottom_right{x_max,y_max},class_index (no space)
                                                /home/minh/keras-yolo3/VOCdevkit/VOC2007/JPEGImages/000012.jpg 156,97,351,270,6
    """
    if load_previous == True and os.path.isfile(data_path):
        data = np.load(data_path)
        print('Loading training data from ' + data_path)
        return data['image_data'], data['box_data'], data['image_shape']
    image_data = []
    box_data = []
    image_shape = []
    with open(annotation_path) as f:
        for line in f.readlines():
            line = line.split(' ')
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            image = Image.open(filename)
            # For the case 2
            # boxed_image, shape_image = letterbox_image(image, tuple(reversed(input_shape)))
            # for the case 1
            boxed_image, shape_image = resize_image(image, tuple(reversed(input_shape)))
            image_data.append(np.array(boxed_image, dtype='uint8'))  # pixel: [0:255] uint8:[-128, 127]
            image_shape.append(np.array(shape_image))

            boxes = np.zeros((max_boxes, 5), dtype='int32')
            # correct the BBs to the image resize
            if len(line)==1:  # if there is no object in this image
                box_data.append(boxes)
            for i, box in enumerate(line[1:]):
                if i < max_boxes:
                    boxes[i] = np.array(list(map(int, box.split(','))))
                else:
                    break
                image_size = np.array(image.size)
                input_size = np.array(input_shape[::-1])
                # for case 2
                # new_size = (image_size * np.min(input_size/image_size)).astype('int32')
                # boxes[:i+1, 0:2] = (boxes[:i+1, 0:2]*new_size/image_size + (input_size-new_size)/2).astype('int32')
                # boxes[:i+1, 2:4] = (boxes[:i+1, 2:4]*new_size/image_size + (input_size-new_size)/2).astype('int32')
                # for case 1
                boxes[:i + 1, 0] = (boxes[:i + 1, 0] * input_size[0] / image_size[0]).astype('int32')
                boxes[:i + 1, 1] = (boxes[:i + 1, 1] * input_size[1] / image_size[1]).astype('int32')
                boxes[:i + 1, 2] = (boxes[:i + 1, 2] * input_size[0] / image_size[0]).astype('int32')
                boxes[:i + 1, 3] = (boxes[:i + 1, 3] * input_size[1] / image_size[1]).astype('int32')
                box_data.append(boxes)
    image_shape = np.array(image_shape)
    image_data = np.array(image_data)
    box_data = np.array(box_data)
    np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape)
    print('Saving training data into ' + data_path)
    return image_data, box_data, image_shape

def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding
    :param: size: input_shape
    :return:boxed_image:
            image_shape: original shape (h, w)
    """
    image_w, image_h = image.size
    image_shape = np.array([image_h, image_w])
    w, h = size
    new_w = int(image_w * min(w/image_w, h/image_h))
    new_h = int(image_h * min(w/image_w, h/image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w-new_w)//2, (h-new_h)//2))
    return boxed_image, image_shape

def resize_image(image, size):
    """
    resize image with changed aspect ratio
    :param image: origin image
    :param size: input_shape
    :return: origin_image_shape + image resize
    """
    image_w, image_h = image.size
    image_shape = np.array([image_h, image_w])
    image_resize = image.resize(size, Image.NEAREST)
    return image_resize, image_shape


def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.constant([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


def preprocess_image(img_path, model_image_size):    
    image = cv2.imread(img_path)
    print(image.shape)
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_AREA)
    # images/dog.jpg use this is good
    #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    h, w, _ = image.shape

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
                
        # colors: RGB, opencv: BGR
        cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(colors[c])), 6)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(reversed(colors[c])), -1)

        cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
    return image
