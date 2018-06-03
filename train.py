from network_function import YOLOv3
from detect_function import predict
from loss_function import compute_loss
from true_boxes_to_y_true import preprocess_true_boxes
from utils.yolo_utils import *
from config import Input_shape, channels, threshold, ignore_thresh
from datetime import datetime
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Get Data #############################################################################################################
# Sua lai cho phu hop voi data cua minh#######
classes_paths = './model_data/boat_classes.txt'  # TODO
classes_data = read_classes(classes_paths)
anchors_paths = './model_data/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)
# get data
annotation_path_train = 'boat_train.txt'
annotation_path_valid = 'boat_valid.txt'
annotation_path_test = 'boat_test.txt'

data_path_train = 'boat_train.npz'
data_path_valid = 'boat_valid.npz'
data_path_test = 'boat_test.npz'

input_shape = (Input_shape, Input_shape)  # multiple of 32
image_data_train, box_data_train, image_shape_train = get_training_data(annotation_path_train, data_path_train,
                                                                        input_shape, max_boxes=100, load_previous=True)
image_data_valid, box_data_valid, image_shape_valid = get_training_data(annotation_path_valid, data_path_valid,
                                                                        input_shape, max_boxes=100, load_previous=True)
image_data_test, box_data_test, image_shape_test = get_training_data(annotation_path_test, data_path_test,
                                                                     input_shape, max_boxes=100, load_previous=True)
number_image_train = np.shape(image_data_train[0])
########################################################################################################################
"""
# Clear the current graph in each run, to avoid variable duplication
# tf.reset_default_graph()
"""
# model_path = "/home/minh/stage/model.ckpt"
# saver = tf.train.Saver()
print("Starting 1st session...")
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# python network_function.py
# tensorboard --logdir="./graphs" --port 6006
# these log file is saved in graphs folder, can delete these older log file
# porte 6006 may be change
# Explicitly create a Graph object
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Start running operations on the Graph.
    # STEP 1: Input data ###############################################################################################
    with tf.name_scope("Input"):
        X = tf.placeholder(tf.float32, shape=[None, Input_shape, Input_shape, 3])  # for image_data
        S = tf.placeholder(tf.float32, shape=[2, ])  # for image shape
        Y = tf.placeholder(tf.float32, shape=[None, 100, 5])  # for box_data
    # Reshape images for visualization
    x_reshape = tf.reshape(X, [-1, Input_shape, Input_shape, 1])
    tf.summary.image("input", x_reshape)
    # STEP 2: Building the graph #######################################################################################
    # Building the graph
    # Generate output tensor targets for filtered bounding boxes.
    scale1, scale2, scale3 = YOLOv3(X, len(classes_paths)).feature_extractor()
    scale_total = []
    scale_total.append(scale1)
    scale_total.append(scale2)
    scale_total.append(scale3)

    with tf.name_scope("Loss"):
        # predict boxes, score, classes
        boxes, scores, classes = predict(scale_total, anchors, len(classes_data), S,
                                         score_threshold=threshold, iou_threshold=ignore_thresh)
        # Label
        y_true = preprocess_true_boxes(Y, 416, anchors, len(classes_data))
        # Calculate loss
        loss = compute_loss(scale_total, y_true, anchors, len(classes_data))
    with tf.name_scope("Optimizer"):
        # optimizer
        learning_rate = 0.001
        decay = 0.0005
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(loss)
    # STEP 3: Build the evaluation step ################################################################################
    # with tf.name_scope("Accuracy"):
    #     # Model evaluation
    #     accuracy = 1  #
    # STEP 4: Merge all summaries for Tensorboard generation ###########################################################
    # create a saver
    saver = tf.train.Saver(tf.global_variables())
    # Build the summary operation based on the TF collection of Summaries
    summary_op = tf.summary.merge_all()

    # STEP 5: Train the model, and write summaries #####################################################################
    with tf.Session(graph=graph) as sess:
        # Summary Writers
        train_summary_writer = tf.summary.FileWriter('./graphs/train', sess.graph)
        validation_summary_writer = tf.summary.FileWriter('./graphs/validation', sess.graph)
        # Build an initialization operation to run below
        # init = tf.global_variables_initializer()  # will be randomly initialized when calling the global initializer.
        # sess.run(init)
        epochs = 30  # consider
        batch_size = 32  # consider
        for epoch in range(epochs):
            start_time = time.time()
            for start, end in zip(range(0, number_image_train, batch_size), range(batch_size, number_image_train+1, batch_size)):
                summary_train, _, loss_value = sess.run([summary_op, loss, optimizer],
                                                        feed_dict={X: (image_data_train[start:end]/255.),
                                                                   S: image_shape_train[start:end],
                                                                   Y: box_data_train[start:end]})
                # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(errors)
                print("epoch: ", epoch + 1, ",\tbatch: ", start, end, ",\tcost: ", loss_value)
                train_summary_writer.add_summary(summary_train, epoch)
            duration = time.time() - start_time

            if (epoch % 10) == 0:  # check accuracy of valid data set each 10 epochs
                num_examples_per_step = np.shape(image_data_train)[0]
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                # Run summaries and measure accuracy on validation set
                summary_valid, loss_valid = sess.run([summary_op, loss],
                                                     feed_dict={X: (image_data_valid/255.),
                                                                S: image_shape_valid,
                                                                Y: box_data_valid})
                validation_summary_writer.add_summary(summary_valid, epoch)
                print(datetime.now(), "epoch", epoch, "accuracy=", 100*loss_valid, "(", examples_per_sec, "examples/sec;", sec_per_batch, "sec/batch)")
            # Saver the model checkpoint periodically
            if (epoch % 10) == 0:
                create_new_folder = "./stage/saver_model"
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass
                checkpoint_path = create_new_folder + "/model" + str(epoch) + ".ckpt"
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % checkpoint_path)
        print("Tuning completed!")
        # Loss in test data set
        summary_test, loss_test = sess.run([summary_op, loss],
                                           feed_dict={X: (image_data_test/255.),
                                                      S: image_shape_test,
                                                      Y: box_data_test})
        print("Loss on test set: ", (100*loss_test))

train_summary_writer.close()
validation_summary_writer.close()
