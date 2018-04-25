import tensorflow as tf 


def init_weights(size, channels, filters):
	w = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1),name='weight')
	return w

def init_biases(filters):
	b = tf.Variable(tf.constant(0.1, shape=[filters]),name='biases')
	return b