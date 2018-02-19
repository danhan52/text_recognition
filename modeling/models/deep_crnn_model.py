import tensorflow as tf
from models.model_builders.cnn_layers import *
from models.model_builders.lstm_layers import *
from models.model_builders.CTC_functions import *
from models.model_builders.create_optimizers import *




def deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size, 
    optimizer="adam", is_training=True):
    # input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
    # labels = tf.placeholder(tf.string, [None])

    alphabet_codes = list(range(len(alphabet)))
    n_classes = len(alphabet)

    # model layers
    cnn_out = deep_cnn(input_tensor, is_training)
    predictions_dict = bidirectional_lstm(cnn_out, n_classes)

    # loss layers
    out = ctc_loss(predictions_dict, labels, input_shape, alphabet, 
        alphabet_codes, batch_size)
    loss_ctc, predictions_dict, CER, accuracy = out

    # optimizer
    train_op = create_optimizer(loss_ctc)

    return train_op, loss_ctc, CER, accuracy, predictions_dict