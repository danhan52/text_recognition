# # Import libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import pickle
import csv
import time

from online_functions.create_ASM_batch import *
from models.deep_crnn_model import *
from models.model_builders.create_dataset import *

mpl.rcParams["figure.figsize"] = (15, 15)


# # Model parameters - only done once

# ** Important editable parameters **

# for now dataset must be one of
# iamHandwriting, BenthamDataset, combined, or ASM
dataset = "ASM"
n_epochs_per_bunch = 3
bunch_size = 100
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

# for getting rid of previous models (to save space)
def remove_old_ckpt(b):
    mdl_base = output_model_dir+"online_model" + b + ".ckpt"
    try:
        os.remove(mdl_base+".data-00000-of-00001")
    except:
        pass
    
    try:
        os.remove(old_restore_nm+".index")
    except:
        pass

    try:
        os.remove(old_restore_nm+".meta")
    except:
        pass
    
    try:
        os.remove(output_model_dir + "online_metrics" + b + ".pkl")

# # Run model - looped

restore_model_nm = input_model_nm
data = pd.DataFrame(columns=["loss", "cer", "accuracy", "labels", "words", "pred", "bunch", "epoch", "batch"])
for b in range(0, data_size, bunch_size):
    # ** create this "bunch" of the dataset **
    create_ASM_batch(b, b+bunch_size-1, "../data")
    
    # ** Load dataset **
    out = create_iterator(csv_file, input_shape, batch_size, False)
    dataset, iterator, next_batch, datasize = out
    n_batches = int(datasize / batch_size)


    print("Starting training...")
    saver = tf.train.Saver()

    data = pd.DataFrame(columns=["loss", "cer", "accuracy", "labels", "words", "pred", "bunch", "epoch", "batch"])

    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver.restore(sess, restore_model_nm)


        writer = tf.summary.FileWriter(output_graph_dir, sess.graph)
        for i in range(n_epochs_per_bunch):
            sess.run(iterator.initializer)      
            print("---------------------------------------------------------")
            print("Starting epoch", i)
            for j in range(0, n_batches):
                input_tensor_b, labels_b = sess.run(next_batch)

                if i < 1: # only predict on first run through
                    # do prediction first
                    pred = "pred"
                    try:
                        cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                     feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                                  "labels":[[labels_b]], "words":[[wordz]], "pred":pred,
                                   "bunch":b, "epoch":i, "batch":j}
                        print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
                    except:
                        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                                  "labels":[[""]], "words":[[""]], "pred":pred,
                                   "bunch":b, "epoch":i, "batch":j}
                        print("Error at ", b, i, j)
                    # save data
                    newdata = pd.DataFrame.from_dict(newdata)
                    data = data.append(newdata)
                    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
                    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")

                # train with new data
                pred = "train"
                try:
                    _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                 feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                              "labels":[[labels_b]], "words":[[wordz]], "pred":pred,
                               "bunch":b, "epoch":i, "batch":j}
                    print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
                except:
                    newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                              "labels":[[""]], "words":[[""]], "pred":pred,
                               "bunch":b, "epoch":i, "batch":j}
                    print("Error at ", b, i, j)
                # save data
                newdata = pd.DataFrame.from_dict(newdata)
                data = data.append(newdata)
                pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
                saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
        restore_model_nm = output_model_dir+"online_model" + str(b) + ".ckpt"
        # delete old files to save space - always keep 2 models
        remove_old_ckpt(str(b-2*bunch_size))

        print('Bunch Finished!') 
        print(" ********************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************")
