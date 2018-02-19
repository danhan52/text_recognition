import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from .simple_layers import *

def bidirectional_lstm(input_tensor, n_classes, list_n_hidden=[256, 256],
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
    predictions_dict = {'prob': lstm_out, 'raw_predictions': raw_pred}

    return predictions_dict