from network_function import *
from detect_function import *
from config import anchors
# TEST BUILD NETWORK
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    path = '/home/minh/stage/model_data/car.jpg'
    width = 416
    height = 416
    image = (Image.open(path))
    M = image
    M = np.asarray(M)
    print("y: ", np.shape(M)[0])
    print("x: ", np.shape(M)[1])
    inputs = image.resize((width, height), Image.NEAREST)
    with tf.name_scope('image'):
        inputs = tf.expand_dims(inputs, axis=0)
        inputs = tf.cast(inputs, dtype=tf.float32)

    a, b, c = YOLOv3(inputs, 80).create()  # build_networks(inputs)
    # a = YOLOv3(inputs, 80).create()
    mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)
    # x, y, z, d = yolo_head(a, anchors[mask[0]], 80, input_shape)

    # x = tf.shape(a)
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    # python network_function.py
    # tensorboard --logdir="./graphs" --port 6006
    # these log file is saved in graphs folder, can delete these older log file
    # porte 6006 may be change
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print(inputs)
        a = sess.run(a)
        # print(sess.run(tf.shape(a)))
        # print(sess.run(a[0][6][6][:5]))
        a = np.squeeze(a, axis=0)
        print(sess.run(tf.shape(a)))
        print(a)
        # print(sess.run(tf.shape(x)))
        fig = plt.figure()
        for i in range(6):
            x = a[..., i]
            plt.subplot(2, 3, i + 1)

            plt.imshow(x)
        fig.suptitle('con_58')
        # fig.savefig('/home/minh/stage/image_graph/hihi/conv_58')
        plt.show()
    writer.close()
    sess.close()
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
import matplotlib.pyplot as plt
from PIL import Image
def test_conv2dlinear():
    path = 'model_data/girl.png'
    witth = 416
    height = 416
    image = (Image.open(path))
    inputs = image.resize((witth, height), Image.NEAREST)
    inputs = tf.expand_dims(inputs, axis=0)
    inputs = tf.cast(inputs, dtype=tf.float32)
    #inputs=tf.random_normal([5, 13, 13, 1024], mean=1, stddev=4, seed = 1)
    a=conv2dlinear(inputs, 3, 255, 1, 1, name="hihi")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(a)
        print(sess.run(tf.shape(a)))
        a = np.squeeze(a, axis=0)
        a = a[..., 100]
        plt.imshow(a)
        plt.show()
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



