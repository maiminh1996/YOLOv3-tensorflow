# MODEL_NETWORK
from charger_poids import W, B
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YOLOv3(object):
    """Implementation of the YOLOv3"""

    def __init__(self, x, num_classes):
        """
        Create the graph ofthe YOLOv3 model
        :param x: Placeholder for the input tensor: (normalised image (416, 416, 3)/255.)
        :param num_classes: Number of classes in the dataset
               if it isn't in the same folder as this code
        """
        self.X = x
        self.NUM_CLASSES = num_classes

    def feature_extractor(self):
        """
        Create the network graph
        :return: feature maps 5+80 in 3 grid (13,13), (26,26), (52, 52)
        """
        print("YOLOv3, let's go!!!!!!!")
        with tf.name_scope("Features"):
            conv_1 = conv2d(self.X, 1, name="conv_1")
            # Downsample#############################################
            conv_2 = conv2d(conv_1, 2, stride=2, name="conv_2")

            conv_3 = conv2d(conv_2, 3, name="conv_3")
            conv_4 = conv2d(conv_3, 4, name="conv_4")
            resn_1 = resnet(conv_2, conv_4, name="resn_1")
            # Downsample#############################################
            conv_5 = conv2d(resn_1, 5, stride=2, name="conv_5")

            conv_6 = conv2d(conv_5, 6, name="conv_6")
            conv_7 = conv2d(conv_6, 7, name="conv_7")
            resn_2 = resnet(conv_5, conv_7, name="resn_2")

            conv_8 = conv2d(resn_2, 8, name="conv_8")
            conv_9 = conv2d(conv_8, 9, name="conv_9")
            resn_3 = resnet(resn_2, conv_9, name="resn_3")
            # Downsample#############################################
            conv_10 = conv2d(resn_3, 10, stride=2, name="conv_10")

            conv_11 = conv2d(conv_10, 11, name="conv_11")
            conv_12 = conv2d(conv_11, 12, name="conv_12")
            resn_4 = resnet(conv_10, conv_12, name="resn_4")

            conv_13 = conv2d(resn_4, 13, name="conv_13")
            conv_14 = conv2d(conv_13, 14, name="conv_14")
            resn_5 = resnet(resn_4, conv_14, name="resn_5")

            conv_15 = conv2d(resn_5, 15, name="conv_15")
            conv_16 = conv2d(conv_15, 16, name="conv_16")
            resn_6 = resnet(resn_5, conv_16, name="resn_6")

            conv_17 = conv2d(resn_6, 17, name="conv_17")
            conv_18 = conv2d(conv_17, 18, name="conv_18")
            resn_7 = resnet(resn_6, conv_18, name="resn_7")

            conv_19 = conv2d(resn_7, 19, name="conv_19")
            conv_20 = conv2d(conv_19, 20, name="conv_20")
            resn_8 = resnet(resn_7, conv_20, name="resn_8")

            conv_21 = conv2d(resn_8, 21, name="conv_21")
            conv_22 = conv2d(conv_21, 22, name="conv_22")
            resn_9 = resnet(resn_8, conv_22, name="resn_9")

            conv_23 = conv2d(resn_9, 23, name="conv_23")
            conv_24 = conv2d(conv_23, 24, name="conv_24")
            resn_10 = resnet(resn_9, conv_24, name="resn_10")

            conv_25 = conv2d(resn_10, 25, name="conv_25")
            conv_26 = conv2d(conv_25, 26, name="conv_26")
            resn_11 = resnet(resn_10, conv_26, name="resn_11")
            # Downsample#############################################
            conv_27 = conv2d(resn_11, 27, stride=2, name="conv_27")

            conv_28 = conv2d(conv_27, 28, name="conv_28")
            conv_29 = conv2d(conv_28, 29, name="conv_29")
            resn_12 = resnet(conv_27, conv_29, name="resn_12")

            conv_30 = conv2d(resn_12, 30, name="conv_30")
            conv_31 = conv2d(conv_30, 31, name="conv_31")
            resn_13 = resnet(resn_12, conv_31, name="resn_13")

            conv_32 = conv2d(resn_13, 32, name="conv_32")
            conv_33 = conv2d(conv_32, 33, name="conv_33")
            resn_14 = resnet(resn_13, conv_33, name="resn_14")

            conv_34 = conv2d(resn_14, 34, name="conv_34")
            conv_35 = conv2d(conv_34, 35, name="conv_35")
            resn_15 = resnet(resn_14, conv_35, name="resn_15")

            conv_36 = conv2d(resn_15, 36, name="conv_36")
            conv_37 = conv2d(conv_36, 37, name="conv_37")
            resn_16 = resnet(resn_15, conv_37, name="resn_16")

            conv_38 = conv2d(resn_16, 38, name="conv_38")
            conv_39 = conv2d(conv_38, 39, name="conv_39")
            resn_17 = resnet(resn_16, conv_39, name="resn_17")

            conv_40 = conv2d(resn_17, 40, name="conv_40")
            conv_41 = conv2d(conv_40, 41, name="conv_41")
            resn_18 = resnet(resn_17, conv_41, name="resn_18")

            conv_42 = conv2d(resn_18, 42, name="conv_42")
            conv_43 = conv2d(conv_42, 43, name="conv_43")
            resn_19 = resnet(resn_18, conv_43, name="resn_19")
            # Downsample##############################################
            conv_44 = conv2d(resn_19, 44, stride=2, name="conv_44")

            conv_45 = conv2d(conv_44, 45, name="conv_45")
            conv_46 = conv2d(conv_45, 46, name="conv_46")
            resn_20 = resnet(conv_44, conv_46, name="resn_20")

            conv_47 = conv2d(resn_20, 47, name="conv_47")
            conv_48 = conv2d(conv_47, 48, name="conv_48")
            resn_21 = resnet(resn_20, conv_48, name="resn_21")

            conv_49 = conv2d(resn_21, 49, name="conv_49")
            conv_50 = conv2d(conv_49, 50, name="conv_50")
            resn_22 = resnet(resn_21, conv_50, name="resn_22")

            conv_51 = conv2d(resn_22, 51, name="conv_51")
            conv_52 = conv2d(conv_51, 52, name="conv_52")
            resn_23 = resnet(resn_22, conv_52, name="resn_23")  # [None, 13,13,1024]
            ##########################################################
        with tf.name_scope('SCALE'):
            with tf.name_scope('scale_1'):
                conv_53 = conv2d(resn_23, 53, name="conv_53")
                conv_54 = conv2d(conv_53, 54, name="conv_54")
                conv_55 = conv2d(conv_54, 55, name="conv_55")  # [None,14,14,512]
                conv_56 = conv2d(conv_55, 56, name="conv_56")
                conv_57 = conv2d(conv_56, 57, name="conv_57")
                conv_58 = conv2d(conv_57, 58, name="conv_58")  # [None,13 ,13,1024]
                conv_59 = conv2d(conv_58, 59, name="conv_59", batch_norm_and_activation=False)
                # [yolo layer] 6,7,8 # 82  --->predict    scale:1, stride:32, detecting large objects => mask: 6,7,8
                # 13x13x255, 255=3*(80+1+4)
            with tf.name_scope('scale_2'):
                route_1 = route1(conv_57, name="route_1")
                conv_60 = conv2d(route_1, 60, name="conv_60")
                upsam_1 = upsample(conv_60, 2, name="upsample_1")
                route_2 = route2(upsam_1, resn_19, name="route_2")
                conv_61 = conv2d(route_2, 61, name="conv_61")
                conv_62 = conv2d(conv_61, 62, name="conv_62")
                conv_63 = conv2d(conv_62, 63, name="conv_63")
                conv_64 = conv2d(conv_63, 64, name="conv_64")
                conv_65 = conv2d(conv_64, 65, name="conv_65")
                conv_66 = conv2d(conv_65, 66, name="conv_66")
                conv_67 = conv2d(conv_66, 67, name="conv_67", batch_norm_and_activation=False)
                # [yolo layer] 3,4,5 # 94  --->predict   scale:2, stride:16, detecting medium objects => mask: 3,4,5
                # 26x26x255, 255=3*(80+1+4)
            with tf.name_scope('scale_3'):
                route_3 = route1(conv_65, name="route_3")
                conv_68 = conv2d(route_3, 68, name="conv_68")
                upsam_2 = upsample(conv_68, 2, name="upsample_2")
                route_4 = route2(upsam_2, resn_11, name="route_4")
                conv_69 = conv2d(route_4, 69, name="conv_69")
                conv_70 = conv2d(conv_69, 70, name="conv_70")
                conv_71 = conv2d(conv_70, 71, name="conv_71")
                conv_72 = conv2d(conv_71, 72, name="conv_72")
                conv_73 = conv2d(conv_72, 73, name="conv_73")
                conv_74 = conv2d(conv_73, 74, name="conv_74")
                conv_75 = conv2d(conv_74, 75, name="conv_75", batch_norm_and_activation=False)
                # [yolo layer] 0,1,2 # 106 --predict scale:3, stride:8, detecting the smaller objects => mask: 0,1,2
                # 52x52x255, 255=3*(80+1+4)
                # Bounding Box:  YOLOv2: 13x13x5
                #                YOLOv3: 13x13x3x3, 3 for each scale

        return conv_59, conv_67, conv_75


# C'est Bon
def conv2d(inputs, idx, name, stride=1, batch_norm_and_activation=True):
    """
    Convolutional layer
    :param inputs:
    :param idx: conv number
    :param stride:
    :param name:
    :param batch_norm_and_activation:
    :return:
    """
    with tf.variable_scope(name):
        weights = tf.Variable(W(idx), dtype=tf.float32, name="weights")
        tf.summary.histogram("weights", weights)  # add summary
        if stride == 2:
            paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
            inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID', name="nn_conv")
        else:
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name="conv")

        if batch_norm_and_activation:
            # conv_1 ---> conv_75 EXCEPT conv_59, conv_67, conv_75
            with tf.variable_scope('BatchNorm'):
                variance_epsilon = tf.constant(0.001, name="epsilon")  # A small float number to avoid dividing by 0
                moving_mean, moving_variance, beta, gamma = B(idx)
                moving_mean = tf.Variable(moving_mean, dtype=tf.float32, name="moving_mean")
                tf.summary.histogram("moving_mean", moving_mean)  # add summary

                moving_variance = tf.Variable(moving_variance, dtype=tf.float32, name="moving_variance")
                tf.summary.histogram("moving_variance", moving_variance)  # add summary

                beta = tf.Variable(beta, dtype=tf.float32, name="beta")
                tf.summary.histogram("beta", beta)  # add summary

                gamma = tf.Variable(gamma, dtype=tf.float32, name="gamma")
                tf.summary.histogram("gamma", gamma)  # add summary

                conv = tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma, variance_epsilon, name='BatchNorm')
            with tf.name_scope('Activation'):
                alpha = tf.constant(0.1, name="alpha")  # Slope of the activation function at x < 0
                acti = tf.maximum(alpha * conv, conv)
            return acti
        else:
            # for conv_59, conv67, conv_75
            biases = tf.Variable(B(idx), dtype=tf.float32, name="biases")
            tf.summary.histogram("biases", biases)  # add summary
            conv = tf.add(conv, biases)
            return conv


# C'est Bon
def resnet(a, b, name):
    """
    :param a: [5, 500, 416, 32]
    :param b: [5, 500, 416, 32]
    :param name: name in graph
    :return: a+b [5, 500, 416, 32]
    """
    with tf.name_scope(name):
        resn = a + b
        return resn


# C'est Bon
def route1(inputs, name):
    """
    :param inputs: [5, 500, 416, 3]
    :param name: name in graph
    :return: output = input [5, 500, 416, 3]
    """
    # [route]-4
    with tf.name_scope(name):
        output = inputs
        return output


# C'est Bon
def route2(input1, input2, name):
    """
    :param input1: [5, 500, 416, 3]
    :param input2: [5, 500, 416, 32]
    :param name: name in graph
    :return: concatenate{input1, input2} [5, 500, 416, 3+32]
             (nối lại)
    """
    # [route]-1, 36
    # [route]-1, 61
    with tf.name_scope(name):
        output = tf.concat([input1, input2], -1, name='concatenate')  # input1:-1, input2: 61
        return output


# C'est Bon
def upsample(inputs, size, name):
    """
    :param inputs: (5, 416, 416, 3) par ex
    :param size: 2 par ex
    :param name: name in graph
    :return: Resize images to size using nearest neighbor interpolation. (5, 832, 832, 3) par ex
    """
    with tf.name_scope(name):
        w = tf.shape(inputs)[1]  # 416
        h = tf.shape(inputs)[2]  # 416
        output = tf.image.resize_nearest_neighbor(inputs, [size * w, size * h])
        return output

"""
def conv2d(inputs, idx, channels, filters, size, stride, name, batch_norm_and_activation=True):
    with tf.variable_scope(name):
        weights = tf.Variable(tf.truncated_normal([size, size, channels, filters], stddev=0.1), name='weights')
        if stride == 2:
            paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
            inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID', name="nn_conv")
        else:
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name="conv")
        if batch_norm_and_activation:
            # conv_1 ---> conv_75 EXCEPT conv_59, conv_67, conv_75
            with tf.variable_scope('BatchNorm'):
                betas = tf.Variable(tf.ones([filters, ], dtype='float32'), name='beta')  # offset
                shift = tf.Variable(tf.zeros([filters, ], dtype='float32'), name='shift')  # gamma or scale
                mean = tf.Variable(tf.ones([filters, ], dtype='float32'), name='moving_mean')
                variance = tf.Variable(tf.ones([filters, ], dtype='float32'), name='moving_variance')
                variance_epsilon = 1e-03  # A small float number to avoid dividing by 0
                conv = tf.nn.batch_normalization(conv, mean, variance, shift,betas,variance_epsilon, name='BatchNorm')
                # moving_mean, moving_variance, beta, gamma = B(idx)
                # conv = tf.nn.batch_normalization(conv, moving_mean, moving_variance,  gamma, beta, variance_epsilon, name='BatchNorm')
            with tf.name_scope('Activation'):
                alpha = 0.1  # Slope of the activation function at x < 0
                return tf.maximum(alpha * conv, conv)
        else:
            # for conv_59, conv67, conv_75
            biases = tf.Variable(tf.constant(0.1, shape=[filters]), name='biases')
            # biases = B(idx)
            conv = tf.add(conv, biases)
            return conv
"""