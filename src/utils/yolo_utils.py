import random
import cv2
import numpy as np
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

def get_training_data(annotation_path, data_path, input_shape, anchors, num_classes, max_boxes=100, load_previous=True):
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
        # return data['image_data'], data['box_data'], data['image_shape']
        return data['image_data'], data['box_data'], data['image_shape'], [data['y_true0'], data['y_true1'], data['y_true2']]
        # return data['image_data'], data['box_data'], data['image_shape'], [data['y_true']]
    image_data = []
    box_data = []
    image_shape = []
    with open(annotation_path) as f:
        GG = f.readlines()
        np.random.shuffle(GG)
        for line in (GG):
            line = line.split(' ')
            filename = line[0]
            if filename[-1] == '\n':
                filename = filename[:-1]
            image = Image.open(filename)
            # For the case 2
            boxed_image, shape_image = letterbox_image(image, tuple(reversed(input_shape)))
            # for the case 1
            # boxed_image, shape_image = resize_image(image, tuple(reversed(input_shape)))
            image_data.append(np.array(boxed_image, dtype=np.uint8))  # pixel: [0:255] uint8:[-128, 127]
            image_shape.append(np.array(shape_image))

            boxes = np.zeros((max_boxes, 5), dtype=np.int32)
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
                new_size = (image_size * np.min(input_size/image_size)).astype(np.int32)
                # Correct BB to new image
                boxes[i:i+1, 0:2] = (boxes[i:i+1, 0:2]*new_size/image_size + (input_size-new_size)/2).astype(np.int32)
                boxes[i:i+1, 2:4] = (boxes[i:i+1, 2:4]*new_size/image_size + (input_size-new_size)/2).astype(np.int32)
                # for case 1
                # boxes[i:i + 1, 0] = (boxes[i:i + 1, 0] * input_size[0] / image_size[0]).astype('int32')
                # boxes[i:i + 1, 1] = (boxes[i:i + 1, 1] * input_size[1] / image_size[1]).astype('int32')
                # boxes[i:i + 1, 2] = (boxes[i:i + 1, 2] * input_size[0] / image_size[0]).astype('int32')
                # boxes[i:i + 1, 3] = (boxes[i:i + 1, 3] * input_size[1] / image_size[1]).astype('int32')
            box_data.append(boxes)
    image_shape = np.array(image_shape)
    image_data = np.array(image_data)
    box_data = (np.array(box_data))
    y_true = preprocess_true_boxes(box_data, input_shape[0], anchors, num_classes)
    # np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape)
    np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape, y_true0=y_true[0], y_true1=y_true[1], y_true2=y_true[2])
    # np.savez(data_path, image_data=image_data, box_data=box_data, image_shape=image_shape, y_true=y_true)
    print('Saving training data into ' + data_path)
    # return image_data, box_data, image_shape
    return image_data, box_data, image_shape, y_true

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

# Partie l'entrainement
def preprocess_true_boxes(true_boxes, Input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    :param true_boxes: array, shape=(N, 100, 5)N:so luong anh,100:so object max trong 1 anh, 5:x_min,y_min,x_max,y_max,class_id
                    Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    :param input_shape: array-like, hw, multiples of 32, shape = (2,)
    :param anchors: array, shape=(9, 2), wh
    :param num_classes: integer
    :return: y_true: list(3 array), shape like yolo_outputs, xywh are reletive value 3 array [N,, 13, 13, 3, 85]
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array([Input_shape, Input_shape], dtype=np.int32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # [m, T, 2]  (x, y)center point of BB
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w = x_max - x_min  [m, T, 2]
                                                            # h = y_max - y_min
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # hw -> wh
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # hw -> wh

    N = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]
    # grid_shapes = [np.array(input_shape // scale, dtype=np.int) for scale in [32, 16, 8]]  # [2,] ---> [3, 2]
    y_true = [np.zeros((N, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + int(num_classes)),
                       dtype=np.float32) for l in range(3)]  # (m, 13, 13, 3, 85)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # [1, 3, 2]
    anchor_maxes = anchors / 2.  # w/2, h/2  [1, 3, 2]
    anchor_mins = -anchor_maxes   # -w/2, -h/2  [1, 3, 2]
    valid_mask = boxes_wh[..., 0] > 0  # w>0 True, w=0 False

    for b in (range(N)):  # for all of N image
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # image 0: wh [[[163., 144.]]]
        # Expand dim to apply broadcasting.
        if len(wh)==0:
            continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        # print("Imageeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee numero:")
        # print("True_boxes numéro %s" %b)
        # print(true_boxes[b][:10])
        for t, n in enumerate(best_anchor):
            for l in range(3):  # 1 in 3 scale
                if n in anchor_mask[l]:  # choose the corresponding mask: best_anchor in [6, 7, 8]or[3, 4, 5]or[0, 1, 2]

                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(np.int32)  #ex: 3+1.2=4.2--> vao ô co y=4
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(np.int32)  # ex: 3+0.5=3.5--> vao o co x=3 --> o (x,y)=(3,4)  # TODO
                    if grid_shapes[l][1]==13 and (i>=13 or j>=13):
                        print(i)
                    # print("object %s------------------------------"%t)
                    # print(grid_shapes[l])
                    # print("  j=%s, i=%s" %(j, i))
                    k = anchor_mask[l].index(n)
                    # print("  scale l:", l, "best anchor k:", k, anchors[:, l + n])

                    c = true_boxes[b, t, 4].astype(np.int32)  # idx classes in voc classes
                    # print(b, c)
                    # print("  idx classes c in voc:", c)
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]  # l: scale; b; idx image; grid(i:y , j:x); k: best anchor; 0:4: (x,y,w,h)/input_shape
                    # print("  x,y,w,h/416:", y_true[l][b, j, i, k, 0:4])
                    y_true[l][b, j, i, k, 4] = 1  # score = 1
                    y_true[l][b, j, i, k, 5 + c] = 1  # classes = 1, the others =0
                    # print("  y_true[l][b, j, i, k, :] with l:%s, b:%s, j:%s, i:%s, k:%s, c:%s" %(l,b,j,i,k,c))
                    # print("  with l:", l, "b:", b, "j:", j, "i:", i, "k:", k, "c:", c)
                    # print(y_true[l][b, j, i, k, :])
                    break  # if chon dung mask (scale) ---> exit (for l in range(3))

    return y_true

# def resize_image(image, size):
#     """
#     resize image with changed aspect ratio
#     :param image: origin image
#     :param size: input_shape
#     :return: origin_image_shape + image resize
#     """
#     image_w, image_h = image.size
#     image_shape = np.array([image_h, image_w])
#     image_resize = image.resize(size, Image.NEAREST)
#     return image_resize, image_shape


# def generate_colors(class_names):
#     import colorsys
#     hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
#     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#     colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
#     random.seed(10101)  # Fixed seed for consistent colors across runs.
#     random.shuffle(colors)  # Shuffle colors to decorate adjacent classes.
#     random.seed(None)  # Reset seed to default.
#     return colors
#
#
# def scale_boxes(boxes, image_shape):
#     """ Scales the predicted boxes in order to be drawable on the image"""
#     import tensorflow as tf
#     height = image_shape[0]
#     width = image_shape[1]
#     image_dims = tf.constant([height, width, height, width])
#     image_dims = tf.reshape(image_dims, [1, 4])
#     boxes = boxes * image_dims
#     return boxes
#
#
# def preprocess_image(img_path, model_image_size):
#     image = cv2.imread(img_path)
#     print(image.shape)
#     resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_AREA)
#     # images/dog.jpg use this is good
#     #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), interpolation=cv2.INTER_CUBIC)
#     resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
#     image_data = np.array(resized_image, dtype='float32')
#     image_data /= 255.
#     image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
#
#     return image, image_data
#
#
# def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
#     h, w, _ = image.shape
#
#     for i, c in reversed(list(enumerate(out_classes))):
#         predicted_class = class_names[c]
#         box = out_boxes[i]
#         score = out_scores[i]
#
#         label = '{} {:.2f}'.format(predicted_class, score)
#
#         top, left, bottom, right = box
#         top = max(0, np.floor(top + 0.5).astype('int32'))
#         left = max(0, np.floor(left + 0.5).astype('int32'))
#         bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
#         right = min(w, np.floor(right + 0.5).astype('int32'))
#         print(label, (left, top), (right, bottom))
#
#         # colors: RGB, opencv: BGR
#         cv2.rectangle(image, (left, top), (right, bottom), tuple(reversed(colors[c])), 6)
#
#         font_face = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#
#         label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
#         label_rect_left, label_rect_top = int(left - 3), int(top - 3)
#         label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
#         cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(reversed(colors[c])), -1)
#
#         cv2.putText(image, label, (left, int(top - 4)), font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
#
#     return image