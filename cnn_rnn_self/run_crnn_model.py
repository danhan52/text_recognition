#!/usr/bin/env python
import argparse
import os
import csv
import time
import numpy as np
# from tqdm import trange
import tensorflow as tf
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

# from .decoding import get_words_from_chars
# from .config import Params, CONST
# from src.model import crnn_fn
# from src.data_handler import data_loader
# from src.data_handler import preprocess_image_for_prediction

# from src.config import Params, Alphabet, import_params_from_json

csv_files_train = "/home/danny/Repos/text_recognition/tf-crnn-master/data/train.csv"
csv_files_eval = "/home/danny/Repos/text_recognition/tf-crnn-master//data/valid.csv"
output_model_dir = "/home/danny/Repos/text_recognition/tf-crnn-master/estimator"
# csv_files_train = "C:/Users/danny/Repos/text_recognition/tf-crnn-master/data/train.csv"
# csv_files_eval = "C:/Users/danny/Repos/text_recognition/tf-crnn-master//data/valid.csv"
# output_model_dir = "C:/Users/danny/Repos/text_recognition/tf-crnn-master/estimator"
n_epochs = 5
gpu = "" # help="GPU 0,1 or '' ", default=''

train_batch_size=64
eval_batch_size=64
learning_rate=1e-3  # 1e-3 recommended
learning_decay_rate=0.95
learning_decay_steps=5000
evaluate_every_epoch=5
save_interval=5e3
input_shape=(117, 1669)
optimizer='adam'
digits_only=False
alphabet=" !\"#&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|~"
alphabet_decoding='same'
alphabet_codes = list(range(len(alphabet)))
n_classes = len(alphabet)
csv_delimiter='\t'

# needed for quickly making convolutional layers
def weightVar(shape, mean=0.0, stddev=0.02, name='weights'):
    init_w = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(init_w, name=name)


def biasVar(shape, value=0.0, name='bias'):
    init_b = tf.constant(value=value, shape=shape)
    return tf.Variable(init_b, name=name)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name=None):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding, name=name)

keep_prob_dropout = 0.7
#input_tensor = features['images']
input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
labels = tf.placeholder(tf.string, [None])
is_training = True

if input_tensor.shape[-1] == 1:
    input_channels = 1
elif input_tensor.shape[-1] == 3:
    input_channels = 3
else:
    raise NotImplementedError

with tf.variable_scope('deep_cnn'):
    # conv1 - maxPool2x2
    with tf.variable_scope('layer1'):
        W = weightVar([3, 3, input_channels, 64])
        b = biasVar([64])
        conv = conv2d(input_tensor, W, name='conv')
        out = tf.nn.bias_add(conv, b)
        conv1 = tf.nn.relu(out)
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool')

    # conv2 - maxPool 2x2
    with tf.variable_scope('layer2'):
        W = weightVar([3, 3, 64, 128])
        b = biasVar([128])
        conv = conv2d(pool1, W)
        out = tf.nn.bias_add(conv, b)
        conv2 = tf.nn.relu(out)
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

    # conv3 - w/batch-norm (as source code, not paper)
    with tf.variable_scope('layer3'):
        W = weightVar([3, 3, 128, 256])
        b = biasVar([256])
        conv = conv2d(pool2, W)
        out = tf.nn.bias_add(conv, b)
        b_norm = tf.layers.batch_normalization(out, axis=-1,
                                               training=is_training, name='batch-norm')
        conv3 = tf.nn.relu(b_norm, name='ReLU')

    # conv4 - maxPool 2x1
    with tf.variable_scope('layer4'):
        W = weightVar([3, 3, 256, 256])
        b = biasVar([256])
        conv = conv2d(conv3, W)
        out = tf.nn.bias_add(conv, b)
        conv4 = tf.nn.relu(out)
        pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                               padding='SAME', name='pool4')

    # conv5 - w/batch-norm
    with tf.variable_scope('layer5'):
        W = weightVar([3, 3, 256, 512])
        b = biasVar([512])
        conv = conv2d(pool4, W)
        out = tf.nn.bias_add(conv, b)
        b_norm = tf.layers.batch_normalization(out, axis=-1,
                                               training=is_training, name='batch-norm')
        conv5 = tf.nn.relu(b_norm)

    # conv6 - maxPool 2x1 (as source code, not paper)
    with tf.variable_scope('layer6'):
        W = weightVar([3, 3, 512, 512])
        b = biasVar([512])
        conv = conv2d(conv5, W)
        out = tf.nn.bias_add(conv, b)
        conv6 = tf.nn.relu(out)
        pool6 = tf.nn.max_pool(conv6, [1, 2, 2, 1], strides=[1, 2, 1, 1],
                               padding='SAME', name='pool6')

    # conv 7 - w/batch-norm (as source code, not paper)
    with tf.variable_scope('layer7'):
        W = weightVar([2, 2, 512, 512])
        b = biasVar([512])
        conv = conv2d(pool6, W, padding='VALID')
        out = tf.nn.bias_add(conv, b)
        b_norm = tf.layers.batch_normalization(out, axis=-1,
                                               training=is_training, name='batch-norm')
        conv7 = tf.nn.relu(b_norm)

    # reshape output
    with tf.variable_scope('Reshaping_cnn'):
        shape = conv7.get_shape().as_list()  # [batch, height, width, features]
        shape_tens = tf.shape(conv7)
        transposed = tf.transpose(conv7, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features]
        conv_out = tf.reshape(transposed, [shape_tens[0], -1, shape[1] * shape[3]],
                                   name='reshaped')  # [batch, width, height x features]

# logprob, raw_pred = deep_bidirectional_lstm(conv, params=parameters, summaries=False)

# def deep_bidirectional_lstm(inputs: tf.Tensor, params: Params, summaries: bool=True) -> tf.Tensor:
# Prepare data shape to match `bidirectional_rnn` function requirements
# Current data input shape: (batch_size, n_steps, n_input) "(batch, time, height)"

list_n_hidden = [256, 256]

with tf.name_scope('deep_bidirectional_lstm'):
    # Forward direction cells
    fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
    # Backward direction cells
    bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

    lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                    bw_cell_list,
                                                                    conv_out, # THE INPUT
                                                                    dtype=tf.float32
                                                                    )

    # Dropout layer
    lstm_net = tf.nn.dropout(lstm_net, keep_prob=keep_prob_dropout)

    with tf.variable_scope('Reshaping_rnn'):
        shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
        rnn_reshaped = tf.reshape(lstm_net, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

    with tf.variable_scope('fully_connected'):
        W = weightVar([list_n_hidden[-1]*2, n_classes])
        b = biasVar([n_classes])
        fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, W), b)

    shape_tens = tf.shape(lstm_net)
    lstm_out = tf.reshape(fc_out, [shape_tens[0], -1, n_classes], name='reshape_out')  # [batch, width, n_classes]

    raw_pred = tf.argmax(tf.nn.softmax(lstm_out), axis=2, name='raw_prediction')

    # Swap batch and time axis
    lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]
    

# Set up for loss and training

# Compute seq_len from image width
n_pools = 2*2  # 2x2 pooling in dimension W on layer 1 and 2
seq_len_inputs = tf.divide([input_shape[1]]*train_batch_size, n_pools,
                           name='seq_len_input_op') - 1

predictions_dict = {'prob': lstm_out,
                    'raw_predictions': raw_pred,
                    }


# Get keys (letters) and values (integer stand ins for letters)
# Alphabet and codes
keys = [c for c in alphabet] # the letters themselves
values = alphabet_codes # integer representations


# Create non-string labels from the keys and values above
# Convert string label to code label
with tf.name_scope('str2code_conversion'):
    table_str2int = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    splited = tf.string_split(labels, delimiter='')  # TODO change string split to utf8 split in next tf version
    codes = table_str2int.lookup(splited.values)
    sparse_code_target = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                 minlength=tf.shape(predictions_dict['prob'])[1])


# Use ctc loss on probabilities from lstm output
# Loss
# ----
# >>> Cannot have longer labels than predictions -> error
with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
    loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                              inputs=predictions_dict['prob'],
                              sequence_length=tf.cast(seq_len_inputs, tf.int32),
                              preprocess_collapse_repeated=False,
                              ctc_merge_repeated=True,
                              ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                              time_major=True)
    loss_ctc = tf.reduce_mean(loss_ctc)
    loss_ctc = tf.Print(loss_ctc, [loss_ctc], message='* Loss : ')

    
# Create the learning rate as well as a moving average
global_step = tf.train.get_or_create_global_step()
# # Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step, zero_debias=True)
# Create the shadow variables, and add op to maintain moving averages
maintain_averages_op = ema.apply([loss_ctc])
loss_ema = ema.average(loss_ctc)

# Train op
# --------
learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           learning_decay_steps, learning_decay_rate,
                                           staircase=True)


# Set up optimizer
if optimizer == 'ada':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
elif optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
elif optimizer == 'rms':
    optimizer = tf.train.RMSPropOptimizer(learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
opt_op = optimizer.minimize(loss_ctc, global_step=global_step)
with tf.control_dependencies(update_ops + [opt_op]):
    train_op = tf.group(maintain_averages_op)

# get the details to make all images the same size
import os
import warnings
import numpy as np
from skimage import io as skimio
from skimage import color as skimcolor
import skimage.transform as skimtrans
import matplotlib.pyplot as plt

local_path = "/home/danny/Repos/text_recognition/tf-crnn-master/"
img_dir = "data/Images_mod/"

with open("./data/train.csv", "r") as f:
    f_lines = f.read().splitlines()
    filenames = [l.split("\t")[0] for l in f_lines]
    label_list = [l.split("\t")[1][1:-1] for l in f_lines]
datasize = len(label_list)
    
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, input_shape)
    return image_decoded, label
    
dataset = tf.data.Dataset.from_tensor_slices((filenames, label_list))
dataset = dataset.map(_parse_function).batch(train_batch_size)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    
    # writer = tf.summary.FileWriter('./my_graph/LogRegNormal', sess.graph)
    n_batches = int(datasize / train_batch_size)
    for i in range(n_epochs):
        print("Starting epoch", i)
        sess.run(iterator.initializer)
        for _ in range(n_batches):
            print("Starting batch")
            input_tensor_b, labels_b = sess.run(next_batch)
            print("Got data")
            tr_batch = sess.run(train_op, feed_dict={input_tensor: input_tensor_b, labels: labels_b})
            print("batch")
            
        print('epoch: {0}, loss: {1}'.format(i, tr_batch))

    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Optimization Finished!') 
