# # Import libraries
import os
import sys
import csv
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from online_functions.run_bunch import *
from models.deep_crnn_model import *
from models.model_builders.create_dataset import *


def run_predict_model(dataset = "BenthamTest", # iamHandwriting, BenthamDataset, BenthamTest, or combined
                      n_epochs = 1,
                      batch_size = 16,
                      input_model_dir = "./tf_output/input_model/",
                      trg = 0):
    
    # ** Less important parameters **
    data_folder = "../data/" + dataset
    csv_file = data_folder + "/train.csv"
    output_model_dir = "./tf_output/estimator/"
    output_graph_dir = "./tf_output/graph/"

    # load input_shape from file output by preprocess
    with open(data_folder + "/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        input_shape = (int(float(h)), int(float(w)))
        print(input_shape)

    # load alphabet from file output by preprocess
    with open(data_folder + "/alphabet.txt", "r") as f:
        alphabet = f.readline()
        print(alphabet)


    # # Run model
    # ** Create model ops **
    input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
    labels = tf.placeholder(tf.string, [None])

    out = deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size,
                    is_training = False)
    train_op, loss_ctc, CER, accuracy, prob, words = out


    # ** Load dataset **
    out = create_iterator(csv_file, input_shape, batch_size, False)
    dataset, iterator, next_batch, datasize = out
    n_batches = int(datasize / batch_size)


    # ** Train model **
    saver = tf.train.Saver()

    try:
        data = pickle.load(open(input_model_dir + "metrics" + str(trg) + ".pkl", "rb"))
        restore_model_nm = input_model_dir + "model" + str(trg) + ".ckpt"
    except:
        data = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                 "loss", "cer", "accuracy", "labels", "words", "filenames", "time"])
        restore_model_nm = input_model_dir + "model" + str(trg) + ".ckpt"

    run_epochs(saver = saver,
               restore_model_nm = restore_model_nm,
               n_epochs_per_bunch = n_epochs,
               iterator = iterator,
               n_batches = n_batches,
               next_batch = next_batch,
               train_op = None,
               CER = CER,
               accuracy = accuracy,
               loss_ctc = loss_ctc,
               words = words,
               input_tensor = input_tensor,
               labels = labels,
               trg = trg+1000,
               data = data,
               output_model_dir = output_model_dir,
               oldnew = "new",
               pred = "pred")

    print("Prediction finished!")
    return

if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_predict_model(dataset = sys.argv[1])
    elif len(sys.argv) == 3:
        run_predict_model(dataset = sys.argv[1],
                          n_epochs = int(sys.argv[2]))
    elif len(sys.argv) == 4:
        run_predict_model(dataset = sys.argv[1],
                          n_epochs = int(sys.argv[2]),
                          batch_size = int(sys.argv[3]))
    elif len(sys.argv) == 5:
        run_predict_model(dataset = sys.argv[1],
                          n_epochs = int(sys.argv[2]),
                          batch_size = int(sys.argv[3]),
                          input_model_dir = sys.argv[4])
    elif len(sys.argv) == 6:
        run_predict_model(dataset = sys.argv[1],
                          n_epochs = int(sys.argv[2]),
                          batch_size = int(sys.argv[3]),
                          input_model_dir = sys.argv[4],
                          trg = int(sys.argv[5]))
    else:
        run_predict_model()