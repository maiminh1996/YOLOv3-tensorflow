# MODEL_NETWORK
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Initialize import init_weights


def conv2dlinear(inputs, channels, filters, size, stride, name):
    with tf.name_scope(name):
        # filters: ouput_filters
        # channels: input_filters
        # channels = tf.shape(inputs) #inputs.get_shape()[3] #get input_filters
        # channels = channels[3]
        weights = init_weights(size, channels, filters)
        # biases = init_biases(filters)
        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 0]])
        inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
        conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID', name="conv")

        activation = conv  # linear
        # conv = tf.add(conv,biases,name="conv_biased")
        tf.summary.histogram('activation', activation)

        return activation

def conv2d(inputs, channels, filters, size, stride, name):
    # convolutional + batch normalization
    with tf.name_scope(name):
        # filters: ouput_filters
        # channels: input_filters
        # channels = tf.shape(input) #inputs.get_shape()[3] #get input_filters
        # channels = channels[3]
        weights = init_weights(size, channels, filters)
        # biases = init_biases(filters)
        if stride > 1:
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name="conv")
        else:
            paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
            inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID', name="conv")

        # batch normalization
        depth = conv.get_shape()[3]

        scale = tf.Variable(np.ones([depth, ], dtype='float32'), name="scale")
        shift = tf.Variable(np.zeros([depth, ], dtype='float32'), name="shift")
        BN_EPSILON = 1e-8
        mean = tf.Variable(np.ones([depth, ], dtype='float32'), name="rolling_mean")
        variance = tf.Variable(np.ones([depth, ], dtype='float32'), name="rolling_variance")

        conv = tf.nn.batch_normalization(conv, mean, variance, shift, scale, BN_EPSILON)
        activation = tf.nn.leaky_relu(conv, alpha=0.1)

        tf.summary.histogram('activation', activation)

        return activation


def resnet(a, b, name):
    with tf.name_scope(name):
        resn = a + b
        return resn


def route1(inputs, name):
    # [route]-4
    # [route]-4
    with tf.name_scope(name):
        output = inputs
        return output


def route2(input1, input2, name):
    # [route]-1, 36
    # [route]-1, 61
    """tf.concat()
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]"""
    with tf.name_scope(name):
        output = tf.concat([input1, input2], -1)  # input1:-1, input2: 61
        return output


def upsample(inputs, stride, name):
    with tf.name_scope(name):
        size1 = tf.shape(inputs)[1]
        size2 = tf.shape(inputs)[2]
        output = tf.image.resize_nearest_neighbor(inputs, [stride * size1, stride * size2])
        # stride=2, size=shape(inputs)[1]
        return output


def build_networks(inputs):
    print("Building YOLOv3 neural network...")
    conv_0 = conv2d(inputs, 3, 32, 3, 1, name="conv_0")  # 0

    # Downsample###############################################################
    conv_1 = conv2d(conv_0, 32, 64, 3, 2, name="conv_1")  # 1
    conv_2 = conv2d(conv_1, 64, 32, 1, 1, name="conv_2")  # 2
    conv_3 = conv2d(conv_2, 32, 64, 3, 1, name="conv_3")  # 3
    resn_0 = resnet(conv_1, conv_3, name="resn_0")  # 4

    # Downsample###############################################################
    conv_4 = conv2d(resn_0, 64, 128, 3, 2, name="conv_4")  # 5
    conv_5 = conv2d(conv_4, 128, 64, 1, 1, name="conv_5")  # 6
    conv_6 = conv2d(conv_5, 64, 128, 3, 1, name="conv_6")  # 7
    resn_1 = resnet(conv_4, conv_6, name="resn_1")  # 8
    conv_7 = conv2d(resn_1, 128, 64, 1, 1, name="conv_7")  # 9
    conv_8 = conv2d(conv_7, 64, 128, 3, 1, name="conv_8")  # 10
    resn_2 = resnet(resn_1, conv_8, name="resn_2")  # 11

    # Downsample################################################################
    conv_9 = conv2d(resn_2, 128, 256, 3, 2, name="conv_9")  # 12
    conv_10 = conv2d(conv_9, 256, 128, 1, 1, name="conv_10")  # 13
    conv_11 = conv2d(conv_10, 128, 256, 3, 1, name="conv_11")  # 14
    resn_3 = resnet(conv_9, conv_11, name="resn_3")  # 15
    conv_12 = conv2d(resn_3, 256, 128, 1, 1, name="conv_12")  # 16
    conv_13 = conv2d(conv_12, 128, 256, 3, 1, name="conv_13")  # 17
    resn_4 = resnet(resn_3, conv_13, name="resn_4")  # 18
    conv_14 = conv2d(resn_4, 256, 128, 1, 1, name="conv_14")  # 19
    conv_15 = conv2d(conv_14, 128, 256, 3, 1, name="conv_15")  # 20
    resn_5 = resnet(resn_4, conv_15, name="resn_5")  # 21
    conv_16 = conv2d(resn_5, 256, 128, 1, 1, name="conv_16")  # 22
    conv_17 = conv2d(conv_16, 128, 256, 3, 1, name="conv_17")  # 23
    resn_6 = resnet(resn_5, conv_17, name="resn_6")  # 24
    conv_18 = conv2d(resn_6, 256, 128, 1, 1, name="conv_18")  # 25
    conv_19 = conv2d(conv_18, 128, 256, 3, 1, name="conv_19")  # 26
    resn_7 = resnet(resn_6, conv_19, name="resn_7")  # 27
    conv_20 = conv2d(resn_7, 256, 128, 1, 1, name="conv_20")  # 28
    conv_21 = conv2d(conv_20, 128, 256, 3, 1, name="conv_21")  # 29
    resn_8 = resnet(resn_7, conv_21, name="resn_8")  # 30
    conv_22 = conv2d(resn_8, 256, 128, 1, 1, name="conv_22")  # 31
    conv_23 = conv2d(conv_22, 128, 256, 3, 1, name="conv_23")  # 32
    resn_9 = resnet(resn_8, conv_23, name="resn_9")  # 33
    conv_24 = conv2d(resn_9, 256, 128, 1, 1, name="conv_24")  # 34
    conv_25 = conv2d(conv_24, 128, 256, 3, 1, name="conv_25")  # 35
    resn_10 = resnet(resn_9, conv_25, name="resn_10")  # 36 ######[ROUTE]-1,36######

    # Downsample################################################################
    conv_26 = conv2d(resn_10, 256, 512, 3, 2, name="conv_26")  # 37
    conv_27 = conv2d(conv_26, 512, 256, 1, 1, name="conv_27")  # 38
    conv_28 = conv2d(conv_27, 256, 512, 3, 1, name="conv_28")  # 39
    resn_11 = resnet(conv_26, conv_28, name="resn_11")  # 40
    conv_29 = conv2d(resn_11, 512, 256, 1, 1, name="conv_29")  # 41
    conv_30 = conv2d(conv_29, 256, 512, 3, 1, name="conv_30")  # 42
    resn_12 = resnet(resn_11, conv_30, name="resn_12")  # 43
    conv_31 = conv2d(resn_12, 512, 256, 1, 1, name="conv_31")  # 44
    conv_32 = conv2d(conv_31, 256, 512, 3, 1, name="conv_32")  # 45
    resn_13 = resnet(resn_12, conv_32, name="resn_13")  # 46
    conv_33 = conv2d(resn_13, 512, 256, 1, 1, name="conv_33")  # 47
    conv_34 = conv2d(conv_33, 256, 512, 3, 1, name="conv_34")  # 48
    resn_14 = resnet(resn_13, conv_34, name="resn_14")  # 49
    conv_35 = conv2d(resn_14, 512, 256, 1, 1, name="conv_35")  # 50
    conv_36 = conv2d(conv_35, 256, 512, 3, 1, name="conv_36")  # 51
    resn_15 = resnet(resn_14, conv_36, name="resn_15")  # 52
    conv_37 = conv2d(resn_15, 512, 256, 1, 1, name="conv_37")  # 53
    conv_38 = conv2d(conv_37, 256, 512, 3, 1, name="conv_38")  # 54
    resn_16 = resnet(resn_15, conv_38, name="resn_16")  # 55
    conv_39 = conv2d(resn_16, 512, 256, 1, 1, name="conv_39")  # 56
    conv_40 = conv2d(conv_39, 256, 512, 3, 1, name="conv_40")  # 57
    resn_17 = resnet(resn_16, conv_40, name="resn_17")  # 58
    conv_41 = conv2d(resn_17, 512, 256, 1, 1, name="conv_41")  # 59
    conv_42 = conv2d(conv_41, 256, 512, 3, 1, name="conv_42")  # 60
    resn_18 = resnet(resn_17, conv_42, name="resn_18")  # 61 ######[ROUTE]-1,61######

    # Downsample################################################################
    conv_43 = conv2d(resn_18, 512, 1024, 3, 2, name="conv_43")  # 62
    conv_44 = conv2d(conv_43, 1024, 512, 1, 1, name="conv_44")  # 63
    conv_45 = conv2d(conv_44, 512, 1024, 3, 1, name="conv_45")  # 64
    resn_19 = resnet(conv_43, conv_45, name="resn_19")  # 65
    conv_46 = conv2d(resn_19, 1024, 512, 1, 1, name="conv_46")  # 66
    conv_47 = conv2d(conv_46, 512, 1024, 3, 1, name="conv_47")  # 67
    resn_20 = resnet(resn_19, conv_47, name="resn_20")  # 68
    conv_48 = conv2d(resn_20, 1024, 512, 1, 1, name="conv_48")  # 69
    conv_49 = conv2d(conv_48, 512, 1024, 3, 1, name="conv_49")  # 70
    resn_21 = resnet(resn_20, conv_49, name="resn_21")  # 71
    conv_50 = conv2d(resn_21, 1024, 512, 1, 1, name="conv_50")  # 72
    conv_51 = conv2d(conv_50, 512, 1024, 3, 1, name="conv_51")  # 73
    resn_22 = resnet(resn_21, conv_51, name="resn_22")  # 74  [None, 13,13,1024]

    ###########################################################################
    conv_52 = conv2d(resn_22, 1024, 512, 1, 1, name="conv_52")  # 75
    conv_53 = conv2d(conv_52, 512, 1024, 3, 1, name="conv_53")  # 76
    conv_54 = conv2d(conv_53, 1024, 512, 1, 1, name="conv_54")  # 77  [None,14,14,512]
    conv_55 = conv2d(conv_54, 512, 1024, 3, 1, name="conv_55")  # 78
    conv_56 = conv2d(conv_55, 1024, 512, 1, 1, name="conv_56")  # 79 ######[ROUTE]-4######
    conv_57 = conv2d(conv_56, 512, 1024, 3, 1, name="conv_57")  # 80  [None,13 ,13,1024]
    conv_58 = conv2dlinear(conv_57, 1024, 255, 1, 1, name="conv_58")  # 81  [None,14,14,255]
    # [yolo layer] 6,7,8 # 82  --->predict       scale:1, stride:32, detecting large objects => mask: 6,7,8
    # 13x13x255, 255=3*(80+1+4)

    route_0 = route1(conv_56, name="route_0")  # 83
    conv_59 = conv2d(route_0, 512, 256, 1, 1, name="conv_59")  # 84
    upsam_0 = upsample(conv_59, 2, name="upsample_0")  # 85
    route_1 = route2(upsam_0, resn_18, name="route_1")  # 86
    conv_60 = conv2d(route_1, 256 + 512, 256, 1, 1, name="conv_60")  # 87
    conv_61 = conv2d(conv_60, 256, 512, 3, 1, name="conv_61")  # 88
    conv_62 = conv2d(conv_61, 512, 256, 1, 1, name="conv_62")  # 89
    conv_63 = conv2d(conv_62, 256, 512, 3, 1, name="conv_63")  # 90
    conv_64 = conv2d(conv_63, 512, 256, 1, 1, name="conv_64")  # 91
    conv_65 = conv2d(conv_64, 256, 512, 3, 1, name="conv_65")  # 92
    conv_66 = conv2dlinear(conv_65, 512, 255, 1, 1, name="conv_66")  # 93
    # [yolo layer] 3,4,5 # 94  --->predict        scale:2, stride:16, detecting medium objects => mask: 3,4,5
    # 26x26x255, 255=3*(80+1+4)

    route_2 = route1(conv_64, name="route_2")  # 95
    conv_67 = conv2d(route_2, 256, 128, 1, 1, name="conv_657")  # 96
    upsam_1 = upsample(conv_67, 2, name="upsample_1")  # 97
    route_3 = route2(upsam_1, resn_10, name="route_3")  # 98
    conv_68 = conv2d(route_3, 128 + 256, 128, 1, 1, name="conv_68")  # 99
    conv_69 = conv2d(conv_68, 128, 256, 3, 1, name="conv_69")  # 100
    conv_70 = conv2d(conv_69, 256, 128, 1, 1, name="conv_70")  # 101
    conv_71 = conv2d(conv_70, 128, 256, 3, 1, name="conv_71")  # 102
    conv_72 = conv2d(conv_71, 256, 128, 1, 1, name="conv_72")  # 103
    conv_73 = conv2d(conv_72, 128, 256, 3, 1, name="conv_73")  # 104
    conv_74 = conv2dlinear(conv_73, 256, 255, 1, 1, name="conv_74")  # 105
    # [yolo layer] 0,1,2 # 106 ---predict         scale:3, stride:8, detecting the smaller objects => mask: 0,1,2
    # 52x52x255, 255=3*(80+1+4)
    # Bounding Box:  YOLOv2: 13x13x5
    #               YOLOv3: 13x13x3x3, 3 for each scale
    return conv_58, conv_66, conv_74


# TEST BUILD NETWORK
def test():
    inputs = tf.random_normal([5, 416, 416, 3], mean=1, stddev=4, seed=1)
    k = tf.shape(inputs)
    a, b, c = build_networks(inputs)
    x, y, z = tf.shape(a), tf.shape(b), tf.shape(c)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("input", sess.run(k))
    print("conv_58", sess.run(x))  # , np.shape(b), np.shape(c))
    print("conv_66", sess.run(y))
    print("conv_74", sess.run(z))
    return 0


test()

"""
#TEST FUNCTION upsample()
def test_upsample():
    inputs=tf.random_normal([5, 416, 488, 3], mean=1, stddev=4, seed = 1)
    a=upsample(inputs,stride=2,name="hihi")
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    b=tf.shape(a)
    x=sess.run(b)
    c=tf.shape(inputs)
    y=sess.run(c)
    print("input", y) #input [  5 416 416   3]
    print("output", x) #output [  5 832 832   3]
    return 0
test_upsample()
"""

"""
#TEST FUNCTION resnet()
def test_resn():
    input1=tf.random_normal([5, 416, 416, 32], mean=1, stddev=4, seed = 1) #axis=-1 nen 5,500,416 phai giong nhau
    input2=tf.random_normal([5, 416, 416, 32], mean=1, stddev=4, seed = 1)
    a=resnet(input1,input2,name="hihi")
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    b=tf.shape(a)
    x=sess.run(b)
    c=tf.shape(input1)
    y=sess.run(c)
    d=tf.shape(input2)
    z=sess.run(d)
    print("input1", y) #input1 [  5 416 416  32]
    print("input2", z) #input2 [  5 416 416  32]
    print("output", x) #output [  5 416 416  32]
    return 0
test_resn()
"""

"""
#TEST FUNCTION route2
def test_route2():
    input1=tf.random_normal([5, 500, 416, 3], mean=1, stddev=4, seed = 1) #axis=-1 nen 5,500,416 phai giong nhau
    input2=tf.random_normal([5, 500, 416, 32], mean=1, stddev=4, seed = 1)
    a=route2(input1,input2,name="hihi")
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    b=tf.shape(a)
    x=sess.run(b)
    c=tf.shape(input1)
    y=sess.run(c)
    d=tf.shape(input2)
    z=sess.run(d)
    print("input1", y) #input1 [  5 500 416   3]
    print("input2", z) #input2 [  5 500 416  32]
    print("output", x) #output [  5 500 416  35]
    return 0
test_route2()
"""

"""
#TEST FUNCTION conv2dlinear()
def test_conv2dlinear():
    inputs=tf.random_normal([5, 13, 13, 1024], mean=1, stddev=4, seed = 1)
    a=conv2dlinear(inputs,1024,255,1,1,name="hihi")
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    b=tf.shape(a)
    x=sess.run(b)
    c=tf.shape(inputs)
    y=sess.run(c)
    print("input", y)  #input [  5 416 416   3]
    print("output", x) #output [  5 415 415  32]
    return 0
test_conv2dlinear()
"""

"""
#TEST FUNCTION conv2d()
def test_conv2d():
    inputs=tf.random_normal([5, 416, 416, 3], mean=1, stddev=4, seed = 1)
    a=conv2d(inputs,3,32,3,1,name="hihi")
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    b=tf.shape(a)
    x=sess.run(b)
    c=tf.shape(inputs)
    y=sess.run(c)
    print("input", y) #input [  5 416 416   3]
    print("output", x) #output [  5 415 415  32]
    return 0
test_conv2d()
"""

"""
def get_classes_names():
    names = []
    with open('coco_classes.txt') as f:
        for name in f.readlines():
            name = name[:-1]
            names.append(name)
    return names
    """

"""
def maxpool2d(self,input,size,stride,name):
    with tf.name_scope(name):
        output = tf.nn.max_pool(input, ksize=[1, size, size, 1], \
                strides=[1, stride, stride, 1], padding='SAME',name="pool")

        return output
        """
