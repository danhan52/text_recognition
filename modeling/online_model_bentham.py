
# coding: utf-8

# # Import libraries

# In[ ]:

import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from models.deep_crnn_model import *
from models.model_builders.create_dataset import *

import sys

# # Model parameters

# ** Important editable parameters **

# In[ ]:

# for now dataset must be one of
# iamHandwriting, BenthamDataset, or combined
dataset = "BenthamDataset"
n_epochs = 1
batch_size = 16
restore = False


# ** Less important parameters **

# In[ ]:

data_folder = "../data/" + dataset
csv_file = data_folder + "/train.csv"
output_model_dir = "./tf_output/estimator/"
input_model_dir = "./tf_output/input_model/"
output_preds_dir = "./tf_output/prediction/"
output_graph_dir = "./tf_output/graph/"

optimizer='adam'
learning_rate=1e-3
learning_decay_rate=0.95
learning_decay_steps=5000

# load input_shape from file output by preprocess
with open("../data/iamHandwriting" + "/img_size.txt", "r") as f:
    doc = f.readline()
    w, h = doc.split(",")
    input_shape = (int(float(h)), int(float(w)))
    print(input_shape)

# load alphabet from file output by preprocess
with open("../data/combined" + "/alphabet.txt", "r") as f:
    alphabet = f.readline()
    print(alphabet)


# # Run model

# ** Create model ops **

# In[ ]:

input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
labels = tf.placeholder(tf.string, [None])

out = deep_crnn(input_tensor, labels, input_shape, alphabet, batch_size, lastlayer=False)
train_op, loss_ctc, CER, accuracy, prob, words = out


# ** Load dataset **

# In[ ]:

out = create_iterator(csv_file, input_shape, batch_size, False)
dataset, iterator, next_batch, datasize = out

# ** Train model **

# In[ ]:
print("Starting training...")
saver = tf.train.Saver()

data = pd.DataFrame(columns=["loss", "cer", "accuracy", "labels", "words", "filenames" "pred", "batch", "round"])

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    saver.restore(sess, input_model_dir+"model9.ckpt")
    
    writer = tf.summary.FileWriter(output_graph_dir, sess.graph)
    n_batches = int(datasize / batch_size)
    for i in range(n_epochs):
        print("---------------------------------------------------------")
        print("Starting epoch", i)
        jump_by = 50
        for j in range(0, n_batches, jump_by):
            sess.run(iterator.initializer)
            for k in range(j):

                input_tensor_b, labels_b, filenames_b = sess.run(next_batch)

                if k > j-jump_by: # only predict on first run through
                    # do prediction first
                    pred = "pred"
                    try:
                        cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                     feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                                  "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                                   "pred":pred, "round":i, "batch":j}
                        print('batch: {0}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(j, cer, acc, loss, k))
                    except:
                        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                                  "labels":[[""]], "words":[[""]], "filenames":[[""]],
                                   "pred":pred, "round":i, "batch":j}
                        print("Error at ", j, k)
                    # save data
                    newdata = pd.DataFrame.from_dict(newdata)
                    data = data.append(newdata)
                    pickle.dump(data, open(output_model_dir+"online_metrics" + str(i) + ".pkl", "wb"))
                    saver.save(sess, output_model_dir+"online_model" + str(i) + ".ckpt")

                # train with new data
                pred = "train"
                try:
                    _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                 feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                              "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                                   "pred":pred, "round":i, "batch":j}
                    print('batch: {0}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(j, cer, acc, loss, k))
                except:
                    newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                              "labels":[[""]], "words":[[""]], "filenames":[[""]],
                                   "pred":pred, "round":i, "batch":j}
                    print("Error at ", j, k)
                    num_errors += 1
                # save data
                newdata = pd.DataFrame.from_dict(newdata)
                data = data.append(newdata)
                pickle.dump(data, open(output_model_dir+"online_metrics" + str(i) + ".pkl", "wb"))
                saver.save(sess, output_model_dir+"online_model" + str(i) + ".ckpt")
                
        print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
        
    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Optimization Finished!') 
