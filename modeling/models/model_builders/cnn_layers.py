import tensorflow as tf
from .simple_layers import *

# create a single convolution given the filter and bias size
def conv2d(input, W_size, b_size, strides=[1, 1, 1, 1], padding='SAME',
            name=None):
    W = weightVar(W_size)
    b = biasVar(b_size)
    conv = tf.nn.conv2d(input, W, strides=strides, padding=padding, name=name)
    out = tf.nn.bias_add(conv, b)
    return out

# create a 7 layer deep CNN with (somewhat) alternating max_pool and
# batch_normalization layers
def deep_cnn(input_tensor, is_training, reshape=True):

    with tf.variable_scope('deep_cnn'):
        # conv1 - maxPool2x2
        with tf.variable_scope('layer1'):
            out = conv2d(input_tensor, [3, 3, 1, 64], [64], name="conv1")
            conv1 = tf.nn.relu(out)
            pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')

        # conv2 - maxPool 2x2
        with tf.variable_scope('layer2'):
            out = conv2d(pool1, [3, 3, 64, 128], [128], name="conv2")
            conv2 = tf.nn.relu(out)
            pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool2')

        # conv3 - w/batch-norm (as source code, not paper)
        with tf.variable_scope('layer3'):
            out = conv2d(pool2, [3, 3, 128, 256], [256], name="conv3")
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv3 = tf.nn.relu(b_norm, name='ReLU')

        # conv4 - maxPool 2x1
        with tf.variable_scope('layer4'):
            out = conv2d(conv3, [3, 3, 256, 256], [256], name="conv4")
            conv4 = tf.nn.relu(out)
            pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                   padding='SAME', name='pool4')

        # conv5 - w/batch-norm
        with tf.variable_scope('layer5'):
            out = conv2d(pool4, [3, 3, 256, 512], [512], name="conv5")
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv5 = tf.nn.relu(b_norm)

        # conv6 - maxPool 2x1 (as source code, not paper)
        with tf.variable_scope('layer6'):
            out = conv2d(conv5, [3, 3, 512, 512], [512], name="conv6")
            conv6 = tf.nn.relu(out)
            pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                                   padding='SAME', name='pool6')

        # conv 7 - w/batch-norm (as source code, not paper)
        with tf.variable_scope('layer7'):
            out = conv2d(pool6, [2, 2, 512, 512], [512], padding="VALID",
                name="conv7")
            b_norm = tf.layers.batch_normalization(out, axis=-1,
                                                   training=is_training, name='batch-norm')
            conv7 = tf.nn.relu(b_norm)

    return conv7