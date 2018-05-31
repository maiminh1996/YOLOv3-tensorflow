# Fonction de coût
from detect_function import yolo_head
from config import Input_shape
import tensorflow as tf
import keras.backend as K
import numpy as np


def compute_loss(YOLO_outputs, Y_true, anchors, num_classes, ignore_thresh=.5):
    """
        Return yolo_loss tensor
    :param YOLO_outputs: list of 3 sortie of yolo_neural_network, ko phai cua predict
    :param Y_true: list(3 array) [(N,13,13,3,85), (N,26,26,3,85), (N,52,52,3,85)]
    :param anchors: array, shape=(T, 2), wh
    :param num_classes: 80
    :param ignore_thresh:float, the iou threshold whether to ignore object confidence loss
    :return: loss
    """
    yolo_outputs = YOLO_outputs
    y_true = Y_true  # output of preprocess_true_boxes [3, None, 13, 13, 3, 2]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = np.array([416, 416])  # tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, dtype=y_true[0].dtype)
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], dtype=y_true[0].dtype) for l in range(3)]
    loss = 0  # init
    m = tf.shape(yolo_outputs[0])[0]
    m = tf.cast(m, dtype=yolo_outputs[0])

    for l in range(3):  # for 3 sortie
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        pred_xy, pred_wh, pred_confidence, pred_class_probs = yolo_head(yolo_outputs[l],
                                                                        anchors[anchor_mask[l]],
                                                                        num_classes,
                                                                        input_shape)
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)  # [None, 13, 13, 3, 4]

        # Darknet box loss.
        xy_delta = (y_true[l][..., :2] - pred_xy) * grid_shapes[l][::-1]  # x_true-x_pred, y_true-y_pred  #TODO
        wh_delta = tf.log(y_true[l][..., 2:4]) - tf.log(pred_wh)  # TODO
        # Avoid log(0)=-inf.
        wh_delta = K.switch(object_mask, wh_delta, tf.zeros_like(wh_delta))
        box_delta = tf.concat([xy_delta, wh_delta], axis=-1)
        box_delta_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(tf.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_IoU(pred_box[b], true_box)
            best_iou = np.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, dtype=true_box.dtype))
            return b + 1, ignore_mask

        _, ignore_mask = control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        box_loss = object_mask * tf.square(box_delta * box_delta_scale)

        confidence_loss = object_mask * tf.square(1 - pred_confidence) + \
                          (1 - object_mask) * tf.square(0 - pred_confidence) * ignore_mask

        class_loss = object_mask * tf.square(true_class_probs - pred_class_probs)

        loss += np.sum(box_loss) + np.sum(confidence_loss) + np.sum(class_loss)

    loss = loss / tf.cast(m, dtype=loss.dtype)
    tf.summary.scalar("Loss", loss)
    return loss


def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
    :param b2: true box, shape=[None, 13, 13, 3, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3, 1] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b1 = tf.expand_dims(b1, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b1_xy = b1[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2.  # w/2, h/2 shape= (None, 13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half  # x,y: left bottom corner of BB
        b1_maxes = b1_xy + b1_wh_half  # x,y: right top corner of BB
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # w1 * h1 (None, 13, 13, 3, 1)

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b2 = tf.expand_dims(b2, -2)  # shape= (None, 13, 13, 3, 1, 4)
        # b2 = tf.expand_dims(b2, 0)  # shape= (1, None, 13, 13, 3, 4)
        b2_xy = b2[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b2_wh = b2[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b2_wh_half = b2_wh / 2.  # w/2, h/2 shape=(None, 13, 13, 3, 1, 2)
        b2_mins = b2_xy - b2_wh_half  # x,y: left bottom corner of BB
        b2_maxes = b2_xy + b2_wh_half  # x,y: right top corner of BB
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # w2 * h2

    with tf.name_scope('Intersection'):
        """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
        intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # intersection: wi * hi (None, 13, 13, 3, 1)

    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (None, 13, 13, 3, 1)

    return IoU



