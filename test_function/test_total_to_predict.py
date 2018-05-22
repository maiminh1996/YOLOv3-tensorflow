from network_function import *
from detect_function import *
from config import anchors
# TEST BUILD NETWORK
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image


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

    a, b, c = YOLOv3(inputs, 80).create()

    K = []
    K.append(a)
    K.append(b)
    K.append(c)
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    image_shape = tf.shape(M[..., 0])
    x, y, z = predict(K, anchors, 80, image_shape)

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
        print((sess.run(tf.shape(x))))
        a = sess.run(a)
        # print(sess.run(tf.shape(a)))
        # print(sess.run(a[0][6][6][:5]))
        a = np.squeeze(a, axis=0)
        print(sess.run(tf.shape(a)))
        print(sess.run(x))
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

test()