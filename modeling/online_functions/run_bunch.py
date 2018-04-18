import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time


#############################################################################################################
# run epochs
def run_epochs(saver,
               restore_model_nm,
               n_epochs_per_bunch,
               iterator,
               n_batches,
               next_batch,
               train_op,
               CER,
               accuracy,
               loss_ctc,
               words,
               input_tensor,
               labels,
               trg,
               data,
               output_model_dir,
               oldnew,
               pred,
               output_graph_dir=""):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        
        if restore_model_nm != "":
            saver.restore(sess, restore_model_nm)
        
        if output_graph_dir != "":
            writer = tf.summary.FileWriter(output_graph_dir, sess.graph)
        start_time = time.time()
        for i in range(n_epochs_per_bunch):
            sess.run(iterator.initializer)      
            print("---------------------------------------------------------")
            print("Starting epoch", i)
            for j in range(0, n_batches):
                err = False
                input_tensor_b, labels_b, filenames_b = sess.run(next_batch)

                try:
                    if train_op is not None: # do training
                        _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                                            feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    else: # do prediction
                        cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                                        feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    newdata = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":loss,
                               "cer":cer,
                               "accuracy":[[acc]],
                               "labels":[[labels_b]],
                               "words":[[wordz]],
                               "filenames":[[filenames_b]],
                               "time":time.time()-start_time}
                    print('batch: {0}:{1}:{2}, loss: {3} \n\tCER: {4}, accuracy: {5}'.format(trg, i, j, loss, cer, acc))
                except:
                    newdata = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":-1,
                               "cer":-1,
                               "accuracy":[[-1, -1]],
                               "labels":[[labels_b]],
                               "words":[[""]],
                               "filenames":[[filenames_b]],
                               "time":time.time()-start_time}
                    print("Error at {0}:{1}:{2}".format(trg, i, j))
                    err = True
                # save data
                newdata = pd.DataFrame.from_dict(newdata)
                data = data.append(newdata)
                pickle.dump(data, open(output_model_dir+"metrics" + str(trg) + ".pkl", "wb"))
                saver.save(sess, output_model_dir+"model" + str(trg) + ".ckpt")
                #if not err and j > 2: break
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
        if output_graph_dir != "":
            writer.close()
    return data

