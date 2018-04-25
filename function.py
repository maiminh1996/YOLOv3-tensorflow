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

        self.conv_0 = self.conv2d(self.x,32,3,1,name='conv_0')
        self.conv_1 = self.conv2d(self.conv_0,64,3,2,name='conv_1')
        self.conv_2 = self.conv2d(self.conv_1,32,1,1,name='conv_2')
        self.conv_3 = self.conv2d(self.conv_2,64,3,1,name='conv_3')

        self.resn_0 = self.resnet(self.conv_1,self.conv_3,name='resn_0')

        self.conv_4 = self.conv2d(self.resn_0,128,3,2,name='conv_4')
        self.conv_5 = self.conv2d(self.conv_4,64,1,1,name='conv_5')
        self.conv_6 = self.conv2d(self.conv_5,128,3,1,name='conv_6')

        self.resn_1 = self.resnet(self.conv_4,self.conv_6,name='resn_1')

        self.conv_7 = self.conv2d(self.resn_1,64,1,1,name='conv_7')
        self.conv_8 = self.conv2d(self.conv_7,128,3,1,name='conv_8')

        self.resn_2 = self.resnet(self.resn_1,self.conv_8,name='resn_2')

        self.conv_9 = self.conv2d(self.resn_2,256,3,2,name='conv_9')
        self.conv_10 = self.conv2d(self.conv_9,128,1,1,name='conv_10')
        self.conv_11 = self.conv2d(self.conv_10,256,3,1,name='conv_11')

        self.resn_3 = self.resnet(self.conv_9,self.conv_11,name='resn_3')

        self.conv_12 = self.conv2d(self.resn_3,128,1,1,name='conv_12')
        self.conv_17 = self.conv2d(self.conv_16,256,3,1)

        self.resn_18 = self.resnet(self.resn_15,self.conv_17)

        self.conv_19 = self.conv2d(self.resn_18,128,1,1)
        self.conv_20 = self.conv2d(self.conv_19,256,3,1)

        self.resn_21 = self.resnet(self.resn_18,self.conv_20)

        self.conv_22 = self.conv2d(self.resn_21,128,1,1)
        self.conv_23 = self.conv2d(self.conv_22,256,3,1)

        self.resn_24 = self.resnet(self.resn_21,self.conv_23)

        self.conv_25 = self.conv2d(self.resn_24,128,1,1)
        self.conv_26 = self.conv2d(self.conv_25,256,3,1)

        self.resn_27 = self.resnet(self.resn_24,self.conv_26)

        self.conv_28 = self.conv2d(self.resn_27,128,1,1)
        self.conv_29 = self.conv2d(self.conv_28,256,3,1)

        self.resn_30 = self.resnet(self.resn_27,self.conv_29)

        self.conv_31 = self.conv2d(self.resn_30,128,1,1)
        self.conv_32 = self.conv2d(self.conv_31,256,3,1)

        self.resn_33 = self.resnet(self.resn_30,self.conv_32)

        self.conv_34 = self.conv2d(self.resn_33,128,1,1)
        self.conv_35 = self.conv2d(self.conv_34,256,3,1)

        self.resn_36 = self.resnet(self.resn_33,self.conv_35)

        self.conv_37 = self.conv2d(self.resn_36,512,3,2)
        self.conv_38 = self.conv2d(self.conv_37,256,1,1)
        self.conv_39 = self.conv2d(self.conv_38,512,3,1)

        self.resn_40 = self.resnet(self.conv_37,self.conv_39)

        self.conv_41 = self.conv2d(self.resn_40,256,1,1)
        self.conv_42 = self.conv2d(self.conv_41,512,3,1)

        self.resn_43 = self.resnet(self.resn_40,self.conv_42)

        self.conv_44 = self.conv2d(self.resn_43,256,1,1)
        self.conv_45 = self.conv2d(self.conv_44,512,3,1)

        self.resn_46 = self.resnet(self.resn_43,self.conv_45)

        self.conv_47 = self.conv2d(self.resn_46,256,1,1)
        self.conv_48 = self.conv2d(self.conv_47,512,3,1)

        self.resn_49 = self.resnet(self.resn_46,self.conv_48)

        self.conv_50 = self.conv2d(self.resn_49,256,1,1)
        self.conv_51 = self.conv2d(self.conv_50,512,3,1)

        self.resn_52 = self.resnet(self.resn_49,self.conv_51)

        self.conv_53 = self.conv2d(self.resn_52,256,1,1)
        self.conv_54 = self.conv2d(self.conv_53,512,3,1)

        self.resn_55 = self.resnet(self.resn_52,self.conv_54)

        self.conv_56 = self.conv2d(self.resn_55,256,1,1)
        self.conv_57 = self.conv2d(self.conv_56,512,3,1)

        self.resn_58 = self.resnet(self.resn_55,self.conv_57)

        self.conv_59 = self.conv2d(self.resn_58,256,1,1)
        self.conv_60 = self.conv2d(self.conv_59,512,3,1)

        self.resn_61 = self.resnet(self.resn_58,self.conv_60)

        self.conv_62 = self.conv2d(self.resn_61,1024,3,2)
        self.conv_63 = self.conv2d(self.conv_62,512,1,1)
        self.conv_64 = self.conv2d(self.conv_63,1024,3,1)

        self.resn_65 = self.resnet(self.conv_62,self.conv_64)

        self.conv_66 = self.conv2d(self.resn_65,512,1,1)
        self.conv_67 = self.conv2d(self.conv_66,1024,3,1)

        self.resn_68 = self.resnet(self.resn_65,self.conv_67)

        self.conv_69 = self.conv2d(self.resn_68,512,1,1)
        self.conv_70 = self.conv2d(self.conv_69,1024,3,1)

        self.resn_71 = self.resnet(self.resn_68,self.conv_70)

        self.conv_72 = self.conv2d(self.resn_71,512,1,1)
        self.conv_73 = self.conv2d(self.conv_72,1024,3,1)
        
        self.resn_74 = self.resnet(self.resn_71,self.conv_73)

        self.conv_75 = self.conv2d(self.resn_74,512,1,1)
        self.conv_76 = self.conv2d(self.conv_75,1024,3,1)
        self.conv_77 = self.conv2d(self.conv_76,512,1,1)
        self.conv_78 = self.conv2d(self.conv_77,1024,3,1)
        self.conv_79 = self.conv2d(self.conv_78,512,1,1)
        self.conv_80 = self.conv2d(self.conv_79,1024,3,1)

        self.conv_81 = self.conv2dlinear(self.conv_80,255,1,1)###################255???

        





        self.boxes, self.probs = self.output_layer(self.conv_30)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.variable_to_restore = tf.global_variables()

        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver.restore(self.sess, self.weights_file)

        print ("Loading complete!" + '\n')


    def conv2dlinear(self,idx,inputs,filters,size,stride,name):
        with tf.name_scope(name):
            #filters: ouput_filters
            channels = inputs.get_shape()[3] #get input_filters 

            weights = init_weights(size, channels, filters)
            biases = init_biases(filters)

            paddings = tf.constant([[1, 0], [1, 0]])
            inputs_pad=tf.pad(input, paddings, "CONSTANT")
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
                conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME',name="conv")
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

    def maxpool2d(self,inputs,size,stride,name):
        with tf.name_scope(name):
            returnn = tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name="pool")

            return returnn

    def output_layer(self,predicts,name):
        with tf.name_scope(name):


    def softmax(self,logits):
        softm=tf.exp(logits)/tf.reduce_sum(tf.exp(logits),axis=0)
        return softm

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)