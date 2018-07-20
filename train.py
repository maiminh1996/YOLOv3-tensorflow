from config import Input_shape, channels, path
from network_function import YOLOv3

from loss_function import compute_loss
from utils.yolo_utils import get_training_data, read_anchors, read_classes

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

np.random.seed(101)

# Add argument
def argument():
    parser = argparse.ArgumentParser(description='COCO or VOC or boat')
    parser.add_argument('--COCO', action='store_true', help='COCO flag')
    parser.add_argument('--VOC', action='store_true', help='VOC flag')
    parser.add_argument('--boat', action='store_true', help='boat flag')
    args = parser.parse_args()
    return args
# Get Data #############################################################################################################
PATH = path + '/yolo3'
classes_paths = PATH + '/model/boat_classes.txt'
classes_data = read_classes(classes_paths)
anchors_paths = PATH + '/model/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)

annotation_path_train = PATH + '/model/boat_train.txt'
annotation_path_valid = PATH + '/model/boat_valid.txt'
annotation_path_test = PATH + '/model/boat_test.txt'

data_path_train = PATH + '/model/boat_train.npz'
data_path_valid = PATH + '/model/boat_valid.npz'
data_path_test = PATH + '/model/boat_test.npz'
VOC = False
args = argument()
if args.VOC == True:
    VOC = True
    classes_paths = PATH + '/model/voc_classes.txt'
    classes_data = read_classes(classes_paths)
    annotation_path_train = PATH + '/model/voc_train.txt'
    annotation_path_valid = PATH + '/model/voc_val.txt'
    # annotation_path_test = PATH + '/model/voc_test.txt'

    data_path_train = PATH + '/model/voc_train.npz'
    data_path_valid = PATH + '/model/voc_valid.npz'
    # data_path_test = PATH + '/model/voc_test.npz'



input_shape = (Input_shape, Input_shape)  # multiple of 32
x_train, box_data_train, image_shape_train, y_train = get_training_data(annotation_path_train, data_path_train,
                                                                        input_shape, anchors, num_classes=len(classes_data), max_boxes=20, load_previous=True)
x_valid, box_data_valid, image_shape_valid, y_valid = get_training_data(annotation_path_valid, data_path_valid,
                                                                        input_shape, anchors, num_classes=len(classes_data), max_boxes=20, load_previous=True)
x_test, box_data_test, image_shape_test, y_test = get_training_data(annotation_path_test, data_path_test,
                                                                    input_shape, anchors, num_classes=len(classes_data), max_boxes=20, load_previous=True)
number_image_train = np.shape(x_train)[0]
print("number_image_train", number_image_train)
number_image_valid = np.shape(x_valid)[0]
print("number_image_valid", number_image_valid)
number_image_test = np.shape(x_test)[0]
print("number_image_test", number_image_test)

# import matplotlib.pyplot as plt
#
# plt.imshow(np.array(x_train[0]))
# plt.show()
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
    scale1, scale2, scale3 = YOLOv3(X, len(classes_data)).feature_extractor()
    scale_total = [scale1, scale2, scale3]

    with tf.name_scope("Loss_and_Detect"):
        # Label
        y_predict = [Y1, Y2, Y3]
        # Calculate loss
        loss = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
        # loss_print = compute_loss(scale_total, y_predict, anchors, len(classes_data), print_loss=False)
        tf.summary.scalar("Loss", loss)
    with tf.name_scope("Optimizer"):
        # optimizer
        # for VOC: lr:0.0001, decay:0.0003 with RMSProOp after 60 epochs
        # learning_rate = tf.placeholder(tf.float32, shape=[1], name='lr')
        decay = 0.0003
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)
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
    saver = tf.train.Saver()
    # Build the summary operation based on the TF collection of Summaries
    # summary_op = tf.summary.merge_all()

    # STEP 5: Train the model, and write summaries #####################################################################
    # The Graph to be launched (described above)
    # config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True) #, gpu_options.allow_growth = False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    with tf.Session(config=config, graph=graph) as sess:
        # Merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # Summary Writers
        # tensorboard --logdir='./graphs/' --port 6005
        if VOC==True:
            train_summary_writer = tf.summary.FileWriter(PATH + '/graphs_VOC1/train', sess.graph)
            validation_summary_writer = tf.summary.FileWriter(PATH + '/graphs_VOC1/validation', sess.graph)
        else:
            train_summary_writer = tf.summary.FileWriter(PATH + '/graphs_boat10/train', sess.graph)
            validation_summary_writer = tf.summary.FileWriter(PATH + '/graphs_boat10/validation', sess.graph)
        # summary_writer = tf.summary.FileWriter('./graphs', sess.graph)
        sess.run(tf.global_variables_initializer())
        # If you want to continue training from check point
        # checkpoint = "/home/minh/PycharmProjects/yolo3/save_model/SAVER_MODEL_boatM/model.ckpt-" + "1"
        # saver.restore(sess, checkpoint)
        epochs = 50  #
        batch_size = 32  # consider
        best_loss_valid = 10e6
        for epoch in range(epochs):
            start_time = time.time()
            # nbr_iteration = epochs * round((12-0)/batch_size)

            ## Training#################################################################################################
            mean_loss_train = []
            #for (start, end) in (zip((range(0, number_image_train, (batch_size))), (range(batch_size, number_image_train+1, batch_size)))):
            for start in (range(0, number_image_train, batch_size)):
                end = start + batch_size
                summary_train, loss_train, _ = sess.run([summary_op, loss, optimizer],
                                                        feed_dict={X: (x_train[start:end]/255.),
                                                                   Y1: y_train[0][start:end],
                                                                   Y2: y_train[1][start:end],
                                                                   Y3: y_train[2][start:end]})  # , options=run_options)
                train_summary_writer.add_summary(summary_train, epoch)
                # Flushes the event file to disk
                train_summary_writer.flush()
                # summary_writer.add_summary(summary_train, counter)
                mean_loss_train.append(loss_train)
                print("(start: %s end: %s, \tepoch: %s)\tloss: %s " %(start, end, epoch + 1, loss_train))

            # summary_writer.add_summary(summary_train, global_step=epoch)
            mean_loss_train = np.mean(mean_loss_train)
            duration = time.time() - start_time
            examples_per_sec = number_image_train / duration
            sec_per_batch = float(duration)
            # Validation ###############################################################################################
            mean_loss_valid = []
            for start in (range(0, number_image_valid, batch_size)):
                end = start + batch_size
                # Run summaries and measure accuracy on validation set
                summary_valid, loss_valid = sess.run([summary_op, loss],
                                                    feed_dict={X: (x_valid[start:end]/255.),
                                                               Y1: y_valid[0][start:end],
                                                               Y2: y_valid[1][start:end],
                                                               Y3: y_valid[2][start:end]})  # ,options=run_options)
                validation_summary_writer.add_summary(summary_valid, epoch)
                # Flushes the event file to disk
                validation_summary_writer.flush()
                mean_loss_valid.append(loss_valid)
            mean_loss_valid = np.mean(mean_loss_valid)
            print("epoch %s / %s \ttrain_loss: %s,\tvalid_loss: %s" %(epoch+1, epochs, mean_loss_train, mean_loss_valid))

            if best_loss_valid > mean_loss_valid:
                best_loss_valid = mean_loss_valid
                if VOC ==True:
                    create_new_folder = PATH + "/save_model/SAVER_MODEL_VOC1"
                else:
                    create_new_folder = PATH + "/save_model/SAVER_MODEL_boat10"
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass
                checkpoint_path = create_new_folder + "/model.ckpt"
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % checkpoint_path)

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
        train_summary_writer.close()
        validation_summary_writer.close()
        # summary_writer.close()



