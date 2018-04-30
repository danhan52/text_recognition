# # Import libraries
import sys
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from run_bunch import *
from models.deep_crnn_model import *
from models.model_builders.create_dataset import *


def run_model(pred_train = "train",
              dataset = "iamHandwriting", # iamHandwriting, BenthamDataset, or combined
              n_epochs = 5,
              batch_size = 16,
              randomize = True,
              trg = 0,
              output_model_dir = "./tf_output/estimator/",
              oldnew = "new",
              input_model_dir = "",
              input_trg = "0"):
    is_training = pred_train == "train"
    csv_file = "../data/" + dataset + "/train.csv"
    
    # load input_shape from file output by preprocess
    with open("../data/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        input_shape = (int(float(h)), int(float(w)))
        print(input_shape)

    # load alphabet from file output by preprocess
    with open("../data/alphabet.txt", "r") as f:
        alphabet = f.readline()
        print(alphabet)


    # # Run model
    # ** Create model ops **
    in_t_shape = [None, input_shape[0], input_shape[1], 1]
    input_tensor = tf.placeholder(tf.float32, in_t_shape)
    labels = tf.placeholder(tf.string, [None])

    out = deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size,
                    is_training=True)
    train_op, loss_ctc, CER, accuracy, prob, words, pred_score = out
    
    if not is_training:
        train_op = None

    # ** Load dataset **
    out = create_iterator(csv_file, input_shape, batch_size, randomize)
    dataset, iterator, next_batch, datasize = out
    n_batches = int(datasize / batch_size)


    # ** Train model **
    saver = tf.train.Saver()
    
    if input_model_dir != "":
        try:
            data_batch = pd.read_csv(input_model_dir + "metrics_batch" + input_trg + ".csv")
            data_image = pd.read_csv(input_model_dir + "metrics_image" + input_trg + ".csv")
        except:
            data_batch = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                               "loss", "cer", "accuracy", "time"])
            data_image = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                               "labels", "words", "pred_score", "filenames"])
        restore_model_nm = input_model_dir + "model" + input_trg + ".ckpt"
    else:
        data_batch = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                           "loss", "cer", "accuracy", "time"])
        data_image = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                           "labels", "words", "pred_score", "filenames"])
        restore_model_nm = ""
    print(restore_model_nm)
    print("Model prepped, now running " + pred_train)
    run_epochs(saver = saver,
               restore_model_nm = restore_model_nm,
               n_epochs_per_bunch = n_epochs,
               iterator = iterator,
               n_batches = n_batches,
               next_batch = next_batch,
               train_op = train_op,
               CER = CER,
               accuracy = accuracy,
               loss_ctc = loss_ctc,
               words = words,
               input_tensor = input_tensor,
               labels = labels,
               trg = trg,
               data_batch = data_batch,
               data_image = data_image,
               output_model_dir = output_model_dir,
               oldnew = oldnew,
               pred = pred_train,
               pred_score = pred_score)

    print("Optimization finished!")
    return

if __name__ == "__main__":
    pred_train = "train"
    dataset = "iamHandwriting"
    n_epochs = 5
    batch_size = 16
    randomize = True
    trg = 0
    output_model_dir = "./tf_output/estimator/"
    oldnew = "new"
    input_model_dir = ""
    input_trg = "0"
    
    for i in range(len(sys.argv)):
        if i == 1:
            pred_train = sys.argv[i]
        elif i == 2:
            dataset = sys.argv[i]
        elif i == 3:
            n_epochs = int(sys.argv[i])
        elif i == 4:
            batch_size = int(sys.argv[i])
        elif i == 5:
            randomize = sys.argv[i] == "True"
        elif i == 6:
            trg = int(sys.argv[i])
        elif i == 7:
            output_model_dir = sys.argv[i]
        elif i == 8:
            oldnew = sys.argv[i]
        elif i == 9:
            input_model_dir = sys.argv[i]
        elif i == 10:
            input_trg = sys.argv[i]
        
    run_model(pred_train = pred_train,
              dataset = dataset, 
              n_epochs = n_epochs,
              batch_size = batch_size,
              randomize = randomize,
              trg = trg,
              output_model_dir = output_model_dir,
              oldnew = oldnew,
              input_model_dir = input_model_dir,
              input_trg = input_trg)
    