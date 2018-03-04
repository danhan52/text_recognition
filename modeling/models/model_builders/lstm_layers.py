import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from .simple_layers import *

def bidirectional_lstm(input_tensor, list_n_hidden=[256, 256],
    keep_prob_dropout=0.7):

    with tf.name_scope('deep_bidirectional_lstm'):
        # Forward direction cells
        fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        input_tensor, # THE INPUT
                                                                        dtype=tf.float32
                                                                        )

        # Dropout layer
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=keep_prob_dropout)

    return lstm_net