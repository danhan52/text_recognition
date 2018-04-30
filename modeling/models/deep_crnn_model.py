import tensorflow as tf
from models.model_builders.cnn_layers import *
from models.model_builders.lstm_layers import *
from models.model_builders.CTC_functions import *
from models.model_builders.create_optimizers import *

def reshape_cnn(conv_out):
    with tf.variable_scope('Reshaping_cnn'):
        shape = conv_out.get_shape().as_list()  # [batch, height, width, features]
        shape_tens = tf.shape(conv_out)
        transposed = tf.transpose(conv_out, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features]
        out = tf.reshape(transposed, [shape_tens[0], -1, shape[1] * shape[3]],
                              name='reshaped')  # [batch, width, height x features]
    return out


#######################################################################
def get_predictions(lstm_net, n_classes, list_n_hidden):
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
    prob = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]
    
    return prob #predictions_dict

#######################################################################

def deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size, 
    optimizer="adam", is_training=True, lastlayer=False):

    alphabet_codes = list(range(len(alphabet)))
    n_classes = len(alphabet)

    # model layers
    cnn_out = deep_cnn(input_tensor, is_training)
    cnn_out = reshape_cnn(cnn_out)
    n_hidden = [256, 256]
    #if is_training:
    lstm_out = bidirectional_lstm(cnn_out, n_hidden)
    #else:
    #    lstm_out = bidirectional_lstm(cnn_out, n_hidden, 1.0)
    prob = get_predictions(lstm_out, n_classes, n_hidden)

    # loss layers
    out = ctc_loss(prob, labels, input_shape, alphabet, 
        alphabet_codes, batch_size)
    loss_ctc, words, pred_score, CER, accuracy = out
    words = tf.convert_to_tensor(words)
    
    # optimizer
    train_op = create_optimizer(loss_ctc, lastlayer)

    return train_op, loss_ctc, CER, accuracy, prob, words, pred_score