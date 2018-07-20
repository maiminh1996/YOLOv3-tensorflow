from utils.yolo_utils import get_training_data, read_anchors, read_classes
from network_function import YOLOv3
from loss_function import compute_loss
from config import Input_shape, channels
from datetime import datetime

from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

# Add argument
def argument():
    parser = argparse.ArgumentParser(description='COCO or VOC or boat')
    parser.add_argument('--COCO', action='store_true', help='COCO flag')
    parser.add_argument('--VOC', action='store_true', help='VOC flag')
    parser.add_argument('--boat', action='store_true', help='boat flag')
    args = parser.parse_args()
    return args
# Get Data #############################################################################################################
classes_paths = './model_data/boat_classes.txt'
classes_data = read_classes(classes_paths)
anchors_paths = './model_data/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)

annotation_path_train = './model_data/boat_train.txt'
annotation_path_valid = './model_data/boat_valid.txt'
annotation_path_test = './model_data/boat_test.txt'

data_path_train = './model_data/boat_train.npz'
data_path_valid = './model_data/boat_valid.npz'
data_path_test = './model_data/boat_test.npz'
VOC = False
args = argument()
if args.VOC == True:
    VOC = True
    classes_paths = './model_data/voc_classes.txt'
    classes_data = read_classes(classes_paths)
    annotation_path_train = './model_data/voc_train.txt'
    annotation_path_valid = './model_data/voc_val.txt'
    # annotation_path_test = './model_data/voc_test.txt'

    data_path_train = './model_data/voc_train.npz'
    data_path_valid = './model_data/voc_valid.npz'
    # data_path_test = './model_data/voc_test.npz'



input_shape = (Input_shape, Input_shape)  # multiple of 32
x_train, box_data_train, image_shape_train, y_train = get_training_data(annotation_path_train, data_path_train,
                                                                        input_shape, anchors, num_classes=len(classes_data), max_boxes=100, load_previous=True)

number_image_train = np.shape(x_train)[0]
print("number_image_train", number_image_train)
X_train = x_train[5:6]
Y_train = [y_train[0][5:6], y_train[1][5:6], y_train[2][5:6]]

# print(np.shape(y_train[0]))
# print(np.shape(y_train[1]))
# print(np.shape(y_train[2]))
########################################################################################################################
"""
# Clear the current graph in each run, to avoid variable duplication
# tf.reset_default_graph()
"""
print("Starting 1st session...")
# Explicitly create a Graph object
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Start running operations on the Graph.
    # STEP 1: Input data ###############################################################################################
    with tf.name_scope("Input"):
        X = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, channels], name='Input')  # for image_data
    with tf.name_scope("Target"):
        Y1 = tf.placeholder(tf.float32, shape=[None, Input_shape/32, Input_shape/32, 3, (5+len(classes_data))], name='target_S1')
        Y2 = tf.placeholder(tf.float32, shape=[None, Input_shape/16, Input_shape/16, 3, (5+len(classes_data))], name='target_S2')
        Y3 = tf.placeholder(tf.float32, shape=[None, Input_shape/8, Input_shape/8, 3, (5+len(classes_data))], name='target_S3')
        # Y = tf.placeholder(tf.float32, shape=[None, 100, 5])  # for box_data
    # Reshape images for visualization
    x_reshape = tf.reshape(X, [-1, Input_shape, Input_shape, 1])
    tf.summary.image("input", x_reshape)
    # STEP 2: Building the graph #######################################################################################
    # Building the graph
    # Generate output tensor targets for filtered bounding boxes.
    scale1, scale2, scale3 = YOLOv3(X, len(classes_data), trainable=False).feature_extractor()
    scale_total = [scale1, scale2, scale3]

    with tf.name_scope("Loss"):
        # Label
        y_predict = [Y1, Y2, Y3]
        # Calculate loss
        loss = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=True)
        # loss_print = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
        tf.summary.scalar("Loss", loss)
    with tf.name_scope("Optimizer"):
        # optimizer
        # for VOC: lr:0.0001, decay:0.0003 with RMSProOp after 60 epochs
        learning_rate = 0.0001
        decay = 0.0003
        # learning_rate = 0.0002
        # decay = 0.0005
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(loss)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.01).minimize(loss)
    # STEP 3: Build the evaluation step ################################################################################
    # with tf.name_scope("Accuracy"):
    #     # Model evaluation
    #     accuracy = 1  #
    # STEP 4: Merge all summaries for Tensorboard generation ###########################################################
    # create a saver
    # saver = tf.train.Saver(tf.global_variables())
    # Returns all variables created with trainable=True
    # saver = tf.train.Saver(var_list=tf.trainable_variables())
    # saver = tf.train.Saver()
    # Build the summary operation based on the TF collection of Summaries
    # summary_op = tf.summary.merge_all()

    # STEP 5: Train the model, and write summaries #####################################################################
    # The Graph to be launched (described above)
    # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True) #, gpu_options.allow_growth = False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    with tf.Session(config=config, graph=graph) as sess:
        # Merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # Summary Writers
        # tensorboard --logdir='./graphs/' --port 6005
        # summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        # If you want to continue training from check point
        # checkpoint = "/home/minh/PycharmProjects/yolo3/SAVER_MODEL_VOCK/model.ckpt-" + "3"
        # saver.restore(sess, checkpoint)
        epochs = 1#
        batch_size = 1  # consider
        best_loss_valid = 10e6
        for epoch in range(epochs):
            start_time = time.time()
            # nbr_iteration = epochs * round((12-0)/batch_size)

            ## Training#################################################################################################
            mean_loss_train = []
            #for (start, end) in (zip((range(0, number_image_train, (batch_size))), (range(batch_size, number_image_train+1, batch_size)))):
            for start in (range(0, np.shape(X_train)[0], batch_size)):
                end = start + batch_size
                loss_train = sess.run([loss],
                                      feed_dict={X: (X_train[start:end] / 255.),
                                                 Y1: Y_train[0][start:end],
                                                 Y2: Y_train[1][start:end],
                                                 Y3: Y_train[2][start:end]})  # , options=run_options)
                # train_summary_writer.add_summary(summary_train, epoch)
                # Flushes the event file to disk
                # train_summary_writer.flush()
                # summary_writer.add_summary(summary_train, counter)
                mean_loss_train.append(loss_train)
                print("(start: %s end: %s, \tepoch: %s)\tloss: %s " %(start, end, epoch + 1, loss_train))

            # summary_writer.add_summary(summary_train, global_step=epoch)
            mean_loss_train = np.mean(mean_loss_train)
            duration = time.time() - start_time
            examples_per_sec = number_image_train / duration
            sec_per_batch = float(duration)

        print("Tuning completed!")

        # Testing ######################################################################################################
        # mean_loss_test = []
        # for start in (range(0, 128, batch_size)):
        #     end = start + batch_size
        #     if end > number_image_train:
        #         end = number_image_train
        #     # Loss in test data set
        #     summary_test, loss_test = sess.run([summary_op, loss],
        #                                        feed_dict={X: (x_test[start:end]/255.),
        #                                                   Y1: y_test[0][start:end],
        #                                                   Y2: y_test[1][start:end],
        #                                                   Y3: y_test[2][start:end]})
        #     mean_loss_test.append(mean_loss_test)
        #     # print("Loss on test set: ", (loss_test))
        # mean_loss_test = np.mean(mean_loss_test)
        # print("Mean loss in all of test set: ", mean_loss_test)
        # summary_writer.flush()
        # train_summary_writer.close()
        # validation_summary_writer.close()
        # summary_writer.close()
