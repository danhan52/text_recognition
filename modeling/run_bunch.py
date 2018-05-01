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
               data_batch,
               data_image,
               output_model_dir,
               oldnew,
               pred,
               pred_score):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        
        if restore_model_nm != "":
            saver.restore(sess, restore_model_nm)
        
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
                        _, cer, acc, loss, wordz, pred_s = sess.run([train_op, CER, accuracy, loss_ctc, words, pred_score],
                                                            feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    else: # do prediction
                        cer, acc, loss, wordz, pred_s = sess.run([CER, accuracy, loss_ctc, words, pred_score],
                                                        feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    new_bat = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":loss,
                               "cer":cer,
                               "accuracy":[[acc]],
                               "time":time.time()-start_time}
                    new_img = {"tr_group":[trg for _ in range(len(labels_b))],
                               "oldnew":[oldnew for _ in range(len(labels_b))],
                               "pred":[pred for _ in range(len(labels_b))],
                               "epoch":[i for _ in range(len(labels_b))],
                               "batch":[j for _ in range(len(labels_b))],
                               "labels":[str(ddd, "utf-8") for ddd in labels_b],
                               "words":[str(ddd, "utf-8") for ddd in wordz],
                               "pred_score":pred_s,
                               "filenames":[str(ddd, "utf-8") for ddd in filenames_b]}
                    tim = (time.time()-start_time)/(i*n_batches + j + 1)
                    print('batch: {0}:{1}:{2}, time per batch: {5}\n\tloss: {3}, CER: {4}'.format(trg, i, j, loss, cer, tim), flush=True)
                except:
                    new_bat = {"tr_group":trg,
                               "oldnew":oldnew,
                               "pred":pred,
                               "epoch":i,
                               "batch":j,
                               "loss":-1,
                               "cer":-1,
                               "accuracy":[[-1, -1]],
                               "time":time.time()-start_time}
                    new_img = {"tr_group":[trg for _ in range(len(labels_b))],
                               "oldnew":[oldnew for _ in range(len(labels_b))],
                               "pred":[pred for _ in range(len(labels_b))],
                               "epoch":[i for _ in range(len(labels_b))],
                               "batch":[j for _ in range(len(labels_b))],
                               "labels":[str(ddd, "utf-8") for ddd in labels_b],
                               "words":["" for _ in range(len(labels_b))],
                               "pred_score":[-1 for _ in range(len(labels_b))],
                               "filenames":[str(ddd, "utf-8") for ddd in filenames_b]}
                    print("Error at {0}:{1}:{2}".format(trg, i, j), flush=True)
                    err = True
                # save data
                new_bat = pd.DataFrame.from_dict(new_bat)
                new_img = pd.DataFrame.from_dict(new_img)
                data_batch = data_batch.append(new_bat)
                data_image = data_image.append(new_img)
                data_batch.to_csv(output_model_dir+"metrics_batch" + str(trg) + ".csv", index=False)
                data_image.to_csv(output_model_dir+"metrics_image" + str(trg) + ".csv", index=False)
                saver.save(sess, output_model_dir+"model" + str(trg) + ".ckpt")
                #if not err: break
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
    return data_batch, data_image

