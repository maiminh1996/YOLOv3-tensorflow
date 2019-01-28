import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()

# CREATE A GRID FOR EACH SCALE
name = 'Create_GRID'
with tf.name_scope(name) as scope:

    grid_shape = tf.constant([13, 13])  # height, width ---> grid 13x13 for scale1
    #         (0,0) (1,0) ...   (12,0)                       grid 26x26 for scale2
    #         (0,1) (1,1) ...   (12,1)                       grid 52x52 for scale3
    #          ...              (12,12)
    # In YOLO the height index is the inner most iteration.
    grid_y = tf.range(0, grid_shape[0])  # array([0,1,...,11,12])
    grid_x = tf.range(0, grid_shape[1])
    grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])  # shape=([13,  1,  1,  1])
    grid_x = tf.reshape(grid_x, [1, -1, 1, 1])  # [1, 13, 1, 1]
    grid_y = tf.tile(grid_y, [1, grid_shape[1], 1, 1])  # [13, 1, 1, 1] ---> [13, 13, 1, 1]
    grid_x = tf.tile(grid_x, [grid_shape[0], 1, 1, 1])  # [1, 13, 1, 1] ---> [13, 13, 1, 1]
    grid = tf.concat([grid_x, grid_y], axis=-1)  # shape=[13, 13,  1,  2]
    grid = tf.cast(grid, dtype=tf.float32)  # change type
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(grid))
    print("hihihihihi")
    print(sess.run((grid[::-1])))

writer.close()
