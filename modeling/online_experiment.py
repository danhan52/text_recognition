# # Import libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import pickle
import csv
import time

from online_functions.run_bunch import *
from online_functions.create_ASM_batch import *
from models.deep_crnn_model import *
from models.model_builders.create_dataset import *


# # Model parameters - only done once

# ** Important editable parameters **

# for now dataset must be one of
# iamHandwriting, BenthamDataset, combined, or ASM
dataset = "ASM"
n_epochs_per_bunch = 1
bunch_size = 1000
batch_size = 16


# ** Less important parameters **

data_folder = "../data/" + dataset
csv_file = data_folder + "/train.csv"
output_model_dir = "./tf_output/estimator/"
output_preds_dir = "./tf_output/prediction/"
output_graph_dir = "./tf_output/graph/"
input_model_nm = "./tf_output/input_model/model.ckpt"

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
    
# load data size
with open(data_folder + "/data_size.txt") as f:
    data_size = int(f.readline())
    print(data_size)


# ** Create model ops **
input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
labels = tf.placeholder(tf.string, [None])

out = deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size, lastlayer=False)
train_op, loss_ctc, CER, accuracy, prob, words = out

# I'll have to choose what my data generation function is above
new_data_generator = create_ASM_batch # e.g.
old_data_generator = create_random_batch # e.g.

# # Run model - looped

restore_model_nm = input_model_nm
data = pd.DataFrame(columns=["tr_group", "oldnew", "pred", "epoch", "batch", # location information
                             "loss", "cer", "accuracy", "labels", "words", "filenames"])
for trg in range(0, data_size, bunch_size):
    saver = tf.train.Saver()
    
    # create this training group of the dataset ##########################################################################
#     redo = True
#     while redo:
#         try:
#             new_data_generator(b, b+bunch_size, "../data")
#             redo = False
#         except:
#             redo = True
    # Load dataset
    out = create_iterator(csv_file, input_shape, batch_size, False)
    dataset, iterator, next_batch, datasize = out
    n_batches = int(datasize / batch_size)

    # predict on the current batch of data ##############################################################################
    print("Predict new data")
    data = run_epochs(saver = saver,
                      restore_model_nm = restore_model_nm,
                      n_epochs_per_bunch = 1,
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
                      trg = trg,
                      data = data,
                      output_model_dir = output_model_dir,
                      oldnew = "new",
                      pred = "pred")
            
    # train on the current batch of data ##############################################################################
    print("Train on new data")
    data = run_epochs(saver = saver,
                      restore_model_nm = restore_model_nm,
                      n_epochs_per_bunch = n_epochs_per_bunch,
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
                      data = data,
                      output_model_dir = output_model_dir,
                      oldnew = "new",
                      pred = "train")
    
    # Train the model with old data from ASM, iam, and bentham
    # create the old training group of the dataset ##########################################################################
#     redo = True
#     while redo:
#         try:
#             old_data_generator(b, b+bunch_size, "../data")
#             redo = False
#         except:
#             redo = True
    # Load dataset
    out = create_iterator(csv_file, input_shape, batch_size, True)
    dataset, iterator, next_batch, datasize = out
    n_batches = int(datasize / batch_size)
    
    # train on the current batch of data ##############################################################################
    print("Training with old data")
    data = run_epochs(saver = saver,
                      restore_model_nm = restore_model_nm,
                      n_epochs_per_bunch = n_epochs_per_bunch,
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
                      data = data,
                      output_model_dir = output_model_dir,
                      oldnew = "old",
                      pred = "train")
    
    restore_model_nm = output_model_dir+"online_model" + str(trg) + ".ckpt"

    # delete old files to save space - always keep 2 models
    remove_old_ckpt(str(trg-2*bunch_size), output_model_dir)

    print('Training Group Finished!') 
    print(" ********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************")