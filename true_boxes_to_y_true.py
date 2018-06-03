from detect_function import yolo_head
from config import Input_shape
import tensorflow as tf
import keras.backend as K
import numpy as np


# Partie l'entrainement
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    :param true_boxes: array, shape=(N, 100, 5)N:so luong anh,100:so object max trong 1 anh, 5:x_min,y_min,x_max,y_max,class_id
                    Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    :param input_shape: array-like, hw, multiples of 32, shape = (2,)
    :param anchors: array, shape=(9, 2), wh
    :param num_classes: integer
    :return: y_true: list(3 array), shape like yolo_outputs, xywh are reletive value 3 array [N,, 13, 13, 3, 85]
    """
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array([Input_shape, Input_shape], dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # [m, T, 2]  (x, y)point centre of BB
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w = x_max - x_min  [m, T, 2]
                                                            # h = y_max - y_min
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # hw -> wh TODO normalize,ki la o input shape
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]  # hw -> wh

    N = true_boxes.shape[0]
    grid_shapes = [np.array(input_shape / scale, dtype=np.int) for scale in [32, 16, 8]]  # [2,] ---> [3, 2]
    y_true = [np.zeros((N, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + int(num_classes)),
                       dtype=np.float32) for l in range(3)]  # (m, 13, 13, 3, 85)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # [1, 3, 2]
    anchor_maxes = anchors / 2.  # w/2, h/2  [1, 3, 2]
    anchor_mins = -anchor_maxes   # -w/2, -h/2  [1, 3, 2]
    valid_mask = boxes_wh[..., 0] > 0  # w>0 True, w=0 False

    for b in range(N):  # for all of N image
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # image 0: wh [[[163., 144.]]]
        # Expand dim to apply broadcasting.
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
        for t, n in enumerate(best_anchor):
            for l in range(3):  # 1 in 3 scale
                if n in anchor_mask[l]:  # choose the corresponding mask: best_anchor in [6, 7, 8]or[3, 4, 5]or[0, 1, 2]
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')  #ex: 3+1.2=4.2--> vao ô co y=4
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')  # ex: 3+0.5=3.5--> vao o co x=3 --> o (x,y)=(3,4)
                    n = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')  # idx classes in voc classes
                    y_true[l][b, j, i, n, 0:4] = true_boxes[b, t, 0:4]  # l: scale; b; idx image; grid(i:y , j:x); n: best anchor; 0:4: (x,y,w,h)/input_shape
                    y_true[l][b, j, i, n, 4] = 1  # score = 1
                    y_true[l][b, j, i, n, 5 + c] = 1  # classes = 1, the others =0
                    break  # if chon dung mask (scale) ---> exit (for l in range(3))

    return y_true
