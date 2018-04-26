import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import os 
import shutil
import cfg

from Init_WeightsAndBiases import init_weights, init_biases

def get_classes_names():
    names = []
    with open('coco_names.txt') as f:
        for name in f.readlines():
            name = name[:-1]
            names.append(name)
    return names
    
class YOLOv3_TF:
    def __init__(self,batch_size,input_size,threshold):
        self.weights_file = 'weights/yolov3.ckpt'

        self.alpha = 0.1
        self.threshold = threshold
        self.iou_threshold = 0.5

        self.classes = get_classes_names()
        self.input_size = input_size
        self.num_class = len(self.classes)
        self.boxes_per_cell = 5
        self.cell_size = self.input_size//32
        # self.batch_size = len(self.fromfile)
        self.batch_size = batch_size
        self.debug = {}

        self.anchors = np.array([   ])
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),(self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.build_networks()
                
    

    def build_networks(self):
        print ("Building YOLOv3 graph...")
        self.x = tf.placeholder('float32', [self.batch_size,self.input_size,self.input_size,3] )

        self.conv_0 = self.conv2d(self.x,32,3,1,name="conv_0")                  #0
        #Downsample###############################################################
        self.conv_1 = self.conv2d(self.conv_0,64,3,2,name="conv_1")             #1
        self.conv_2 = self.conv2d(self.conv_1,32,1,1,name="conv_2")             #2 
        self.conv_3 = self.conv2d(self.conv_2,64,3,1,name="conv_3")             #3

        self.resn_0 = self.resnet(self.conv_1,self.conv_3,name="resn_0")        #4
        #Downsample###############################################################
        self.conv_4 = self.conv2d(self.resn_0,128,3,2,name="conv_4")            #5
        self.conv_5 = self.conv2d(self.conv_4,64,1,1,name="conv_5")             #6
        self.conv_6 = self.conv2d(self.conv_5,128,3,1,name="conv_6")            #7

        self.resn_1 = self.resnet(self.conv_4,self.conv_6,name="resn_1")        #8

        self.conv_7 = self.conv2d(self.resn_1,64,1,1,name="conv_7")             #9
        self.conv_8 = self.conv2d(self.conv_7,128,3,1,name="conv_8")            #10

        self.resn_2 = self.resnet(self.resn_1,self.conv_8,name="resn_2")        #11
        #Downsample################################################################
        self.conv_9 = self.conv2d(self.resn_2,256,3,2,name="conv_9")            #12
        self.conv_10 = self.conv2d(self.conv_9,128,1,1,name="conv_10")          #13
        self.conv_11 = self.conv2d(self.conv_10,256,3,1,name="conv_11")         #14

        self.resn_3 = self.resnet(self.conv_9,self.conv_11,name="resn_3")       #15

        self.conv_12 = self.conv2d(self.resn_3,128,1,1,name="conv_12")          #16
        self.conv_13 = self.conv2d(self.conv_12,256,3,1,name="conv_13")         #17

        self.resn_4 = self.resnet(self.resn_3,self.conv_13,name="resn_4")       #18

        self.conv_14 = self.conv2d(self.resn_4,128,1,1,name="conv_14")          #19
        self.conv_15 = self.conv2d(self.conv_14,256,3,1,name="conv_15")         #20

        self.resn_5 = self.resnet(self.resn_4,self.conv_15,name="resn_5")       #21

        self.conv_16 = self.conv2d(self.resn_5,128,1,1,name="conv_16")          #22
        self.conv_17 = self.conv2d(self.conv_16,256,3,1,name="conv_17")         #23

        self.resn_6 = self.resnet(self.resn_5,self.conv_17,name="resn_6")       #24

        self.conv_18 = self.conv2d(self.resn_6,128,1,1,name="conv_18")          #25
        self.conv_19 = self.conv2d(self.conv_18,256,3,1,name="conv_19")         #26

        self.resn_7 = self.resnet(self.resn_6,self.conv_19,name="resn_7")       #27

        self.conv_20 = self.conv2d(self.resn_7,128,1,1,name="conv_20")          #28
        self.conv_21 = self.conv2d(self.conv_20,256,3,1,name="conv_21")         #29

        self.resn_8 = self.resnet(self.resn_7,self.conv_21,name="resn_8")       #30

        self.conv_22 = self.conv2d(self.resn_8,128,1,1,name="conv_22")          #31
        self.conv_23 = self.conv2d(self.conv_22,256,3,1,name="conv_23")         #32

        self.resn_9 = self.resnet(self.resn_8,self.conv_23,name="resn_9")       #33

        self.conv_24 = self.conv2d(self.resn_9,128,1,1,name="conv_24")          #34
        self.conv_25 = self.conv2d(self.conv_24,256,3,1,name="conv_25")         #35

        self.resn_10 = self.resnet(self.resn_9,self.conv_25,name="resn_10")     #36 ######[ROUTE]-1,36######
        #Downsample################################################################
        self.conv_26 = self.conv2d(self.resn_10,512,3,2,name="conv_26")         #37
        self.conv_27 = self.conv2d(self.conv_26,256,1,1,name="conv_27")         #38
        self.conv_28 = self.conv2d(self.conv_27,512,3,1,name="conv_28")         #39

        self.resn_11 = self.resnet(self.conv_26,self.conv_28,name="resn_11")    #40

        self.conv_29 = self.conv2d(self.resn_11,256,1,1,name="conv_29")         #41
        self.conv_30 = self.conv2d(self.conv_29,512,3,1,name="conv_30")         #42

        self.resn_12 = self.resnet(self.resn_11,self.conv_30,name="resn_12")    #43

        self.conv_31 = self.conv2d(self.resn_12,256,1,1,name="conv_31")         #44
        self.conv_32 = self.conv2d(self.conv_31,512,3,1,name="conv_32")         #45

        self.resn_13 = self.resnet(self.resn_12,self.conv_32,name="resn_13")    #46

        self.conv_33 = self.conv2d(self.resn_13,256,1,1,name="conv_33")         #47
        self.conv_34 = self.conv2d(self.conv_33,512,3,1,name="conv_34")         #48

        self.resn_14 = self.resnet(self.resn_13,self.conv_34,name="resn_14")    #49

        self.conv_35 = self.conv2d(self.resn_14,256,1,1,name="conv_35")         #50
        self.conv_36 = self.conv2d(self.conv_35,512,3,1,name="conv_36")         #51

        self.resn_15 = self.resnet(self.resn_14,self.conv_36,name="resn_15")    #52

        self.conv_37 = self.conv2d(self.resn_15,256,1,1,name="conv_37")         #53
        self.conv_38 = self.conv2d(self.conv_37,512,3,1,name="conv_38")         #54

        self.resn_16 = self.resnet(self.resn_15,self.conv_38,name="resn_16")    #55

        self.conv_39 = self.conv2d(self.resn_16,256,1,1,name="conv_39")         #56
        self.conv_40 = self.conv2d(self.conv_39,512,3,1,name="conv_40")         #57

        self.resn_17 = self.resnet(self.resn_16,self.conv_40,name="resn_17")    #58

        self.conv_41 = self.conv2d(self.resn_17,256,1,1,name="conv_41")         #59
        self.conv_42 = self.conv2d(self.conv_41,512,3,1,name="conv_42")         #60

        self.resn_18 = self.resnet(self.resn_17,self.conv_42,name="resn_18")    #61 ######[ROUTE]-1,61######
        #Downsample################################################################
        self.conv_43 = self.conv2d(self.resn_18,1024,3,2,name="conv_43")        #62
        self.conv_44 = self.conv2d(self.conv_43,512,1,1,name="conv_44")         #63
        self.conv_45 = self.conv2d(self.conv_44,1024,3,1,name="conv_45")        #64

        self.resn_19 = self.resnet(self.conv_43,self.conv_45,name="resn_19")    #65

        self.conv_46 = self.conv2d(self.resn_19,512,1,1,name="conv_46")         #66
        self.conv_47 = self.conv2d(self.conv_46,1024,3,1,name="conv_47")        #67

        self.resn_20 = self.resnet(self.resn_19,self.conv_47,name="resn_20")    #68

        self.conv_48 = self.conv2d(self.resn_20,512,1,1,name="conv_48")         #69
        self.conv_49 = self.conv2d(self.conv_48,1024,3,1,name="conv_49")        #70

        self.resn_21 = self.resnet(self.resn_20,self.conv_49,name="resn_21")    #71

        self.conv_50 = self.conv2d(self.resn_21,512,1,1,name="conv_50")         #72
        self.conv_51 = self.conv2d(self.conv_50,1024,3,1,name="conv_51")        #73
        
        self.resn_22 = self.resnet(self.resn_21,self.conv_51,name="resn_22")    #74
        ###########################################################################
        self.conv_52 = self.conv2d(self.resn_22,512,1,1,name="conv_52")         #75
        self.conv_53 = self.conv2d(self.conv_52,1024,3,1,name="conv_53")        #76
        self.conv_54 = self.conv2d(self.conv_53,512,1,1,name="conv_54")         #77
        self.conv_55 = self.conv2d(self.conv_54,1024,3,1,name="conv_55")        #78
        self.conv_56 = self.conv2d(self.conv_55,512,1,1,name="conv_56")         #79 ######[ROUTE]-4######
        self.conv_57 = self.conv2d(self.conv_56,1024,3,1,name="conv_57")        #80
        self.conv_58 = self.conv2dlinear(self.conv_80,255,1,1,name="conv_58")   #81  --->predict
        #[yolo] 6,7,8############################################################82
        self.route_0 = self.route_1(self.conv_56)                               #83
        self.conv_59 = self.conv2d(self.route_0,256,1,1,name="conv_59")         #84
        self.upsam_0 = self.upsample(self.conv_59,2,name="upsample_0")          #85
        self.route_1 = self.route_2(self.upsam_0,self.resn_18,name="route_1")   #86
        self.conv_60 = self.conv2d(self.route_1,256,1,1,name="conv_60")         #87
        self.conv_61 = self.conv2d(self.conv_60,512,3,1,name="conv_61")         #88
        self.conv_62 = self.conv2d(self.conv_61,256,1,1,name="conv_62")         #89
        self.conv_63 = self.conv2d(self.conv_62,512,3,1,name="conv_63")         #90
        self.conv_64 = self.conv2d(self.conv_63,256,1,1,name="conv_64")         #91
        self.conv_65 = self.conv2d(self.conv_64,512,3,1,name="conv_65")         #92
        self.conv_66 = self.conv2dlinear(self.conv_65,255,1,1,name="conv_66")   #93  --->predict
        #[yolo] 3,4,5############################################################94
        self.route_2 = self.route_1(self.conv_64)                               #95
        self.conv_67 = self.conv2d(self.route_2,128,1,1,name="conv_657")        #96
        self.upsam_1 = self.upsample(self.conv_67,2,name="upsample_1    ")      #97
        self.route_3 = self.route_2(self.upsam_1,self.resn_10,name="route_3")   #98
        self.conv_68 = self.conv2d(self.route_3,128,1,1,name="conv_68")         #99
        self.conv_69 = self.conv2d(self.conv_68,256,3,1,name="conv_69")         #100
        self.conv_70 = self.conv2d(self.conv_69,128,1,1,name="conv_70")         #101
        self.conv_71 = self.conv2d(self.conv_70,256,3,1,name="conv_71")         #102
        self.conv_72 = self.conv2d(self.conv_71,128,1,1,name="conv_72")         #103
        self.conv_73 = self.conv2d(self.conv_72,256,3,1,name="conv_73")         #104
        self.conv_74 = self.conv2dlinear(self.route_1,255,1,1,name="conv_74")   #105  ---predict
        #[yolo] 0,1,2############################################################106
        return self.conv_58, self.conv_66, self.conv_74
        
"""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.variable_to_restore = tf.global_variables()

        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver.restore(self.sess, self.weights_file)

        print ("Loading complete!" + '\n')"""


    def conv2dlinear(self,idx,inputs,filters,size,stride,name):
        with tf.name_scope(name):
            #filters: ouput_filters
            channels = inputs.get_shape()[3] #get input_filters 

            weights = init_weights(size, channels, filters)
            biases = init_biases(filters)

            paddings = tf.constant([[1, 0], [1, 0]])
            inputs_pad=tf.pad(self.input, paddings, "CONSTANT")
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID',name="conv")

            activation=conv #linear
            #conv = tf.add(conv,biases,name="conv_biased")  
            tf.summary.histogram('activation', activation)

            return activation

    def conv2d(self,inputs,filters,size,stride,name):
        #convolutional + batch normalization
        with tf.name_scope(name):
            #filters: ouput_filters
            channels = inputs.get_shape()[3] #get input_filters 

            weights = init_weights(size, channels, filters)   
            biases = init_biases(filters)

            if stride>1:
                conv = tf.nn.conv2d(self.inputs, weights, strides=[1, stride, stride, 1], padding='SAME',name="conv")
            else:
                paddings = tf.constant([[1, 0], [1, 0]])
                inputs_pad=tf.pad(input, paddings, "CONSTANT")
                conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID',name="conv")

            # batch normalization
            depth = conv.get_shape()[3]

            scale = tf.Variable(np.ones([depth,], dtype='float32'),name="scale")
            shift = tf.Variable(np.zeros([depth,], dtype='float32'),name="shift")
            BN_EPSILON = 1e-8
            mean = tf.Variable(np.ones([depth,], dtype='float32'),name="rolling_mean")
            variance = tf.Variable(np.ones([depth,], dtype='float32'),name="rolling_variance")

            conv = tf.nn.batch_normalization(conv, mean, variance, shift, scale, BN_EPSILON)
            activation = tf.nn.leaky_relu(conv, alpha=0.1)

            tf.summary.histogram('activation', activation)

            return activation

    def resnet(self,a,b,name):
        with tf.name_scope(name):
            resn=a+b

            return resn

    def maxpool2d(self,input,size,stride,name):
        with tf.name_scope(name):
            output = tf.nn.max_pool(self.input, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name="pool")

            return output

    def output_layer(self,predicts,name):
        with tf.name_scope(name):


    def route_1(self,input,name):
        #[route]-4
        #[route]-4
        with tf.name_scope(name):
            output=input
            return 

    def route_2(self,input1,input2,name):
        #[route]-1, 36
        #[route]-1, 61
        """tf.concat()
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]"""
        with tf.name_scope(name):
            output=tf.concat([self.input1,self.input2], 3) #input1:-1, input2: 61
            return output

    def upsample(self,input,stride,name):
        with tf.name_scope(name):
            tf.image.resize_nearest_neighbor(self.input, [2*stride, 2*stride]) 
    
    def softmax(self,logits):
        softm=tf.exp(logits)/tf.reduce_sum(tf.exp(logits),axis=0)
        return softm

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)