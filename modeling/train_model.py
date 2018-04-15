# # Import libraries
import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from online_functions.run_bunch import *
from models.deep_crnn_model import *
from models.model_builders.create_dataset import *


# # Model parameters
# ** Important editable parameters **
# for now dataset must be one of
# iamHandwriting, BenthamDataset, or combined
dataset = "iamHandwriting"
n_epochs = 5
batch_size = 16


# ** Less important parameters **
data_folder = "../data/" + dataset
csv_file = data_folder + "/train.csv"
output_model_dir = "./tf_output/estimator/"
output_graph_dir = "./tf_output/graph/"
input_model_dir = ""# "./tf_output/input_model/"

optimizer='adam'
learning_rate=1e-3
learning_decay_rate=0.95
learning_decay_steps=5000

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

out = deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size)
train_op, loss_ctc, CER, accuracy, prob, words = out


# ** Load dataset **
out = create_iterator(csv_file, input_shape, batch_size, False)
dataset, iterator, next_batch, datasize = out
n_batches = int(datasize / batch_size)


# ** Train model **
saver = tf.train.Saver()

if input_model_dir != "":
    data = pickle.load(open(input_model_dir+"metrics.pkl", "rb"))
    restore_model_nm = input_model_dir + "model.ckpt"
else:
    data = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                                 "loss", "cer", "accuracy", "labels", "words", "filenames"])
    restore_model_nm = ""

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
           trg = 0,
           data = data,
           output_model_dir = output_model_dir,
           oldnew = "new",
           pred = "train")

print("Optimization finished!")