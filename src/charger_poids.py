from config import path
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def W(number_conv):
    # Charger weights from the pre-trained in COCO
    import h5py
    with h5py.File(path + '/yolo3/model/yolov3.h5', 'r') as f:
        name = 'conv2d_' + str(number_conv)
        w = f['model_weights'][name][name]['kernel:0']
        weights = tf.cast(w, tf.float32)
    return weights


def B(number_conv):
    # Charger biases, bat_norm from the pre-trained in COCO
    import h5py
    with h5py.File(path + '/yolo3/model/yolov3.h5', 'r') as f:
        if (number_conv == 59) or (number_conv == 67) or (number_conv == 75):
            name = 'conv2d_' + str(number_conv)
            b = f['model_weights'][name][name]['bias:0']
            biases = tf.cast(b, tf.float32)
            return biases
        else:
            if 68 <= number_conv <= 74:
                name = 'batch_normalization_' + str(number_conv-2)
                if number_conv==74:
                    print("Finir de charger les poids!")
            elif 66 >= number_conv >= 60:
                name = 'batch_normalization_' + str(number_conv - 1)
            elif 0 < number_conv <= 58:
                name = 'batch_normalization_' + str(number_conv)
            beta = f['model_weights'][name][name]['beta:0']
            beta = tf.cast(beta, tf.float32)

            gamma = f['model_weights'][name][name]['gamma:0']
            gamma = tf.cast(gamma, tf.float32)

            moving_mean = f['model_weights'][name][name]['moving_mean:0']
            moving_mean = tf.cast(moving_mean, tf.float32)

            moving_variance = f['model_weights'][name][name]['moving_variance:0']
            moving_variance = tf.cast(moving_variance, tf.float32)

            return moving_mean, moving_variance, beta, gamma
"""
with tf.Session() as sess:

    a,b,c,d = B(74)
    print(sess.run((a)))
    print(sess.run(tf.shape(a)))
    print(sess.run((b)))
    print(sess.run(tf.shape(b)))
    print(sess.run(c))
    print(sess.run(tf.shape(c)))
    # w = W(75)
    # print(sess.run(w))
    # print(sess.run(tf.shape(w)))
    # d = B(75)
    # print(sess.run(d))
    # print(sess.run(tf.shape(d)))
    sess.close()
"""