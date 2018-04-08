import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import os

# for getting rid of previous models (to save space)
def remove_old_ckpt(b, output_model_dir):
    mdl_base = output_model_dir+"online_model" + b + ".ckpt"
    try:
        os.remove(mdl_base+".data-00000-of-00001")
    except:
        pass
    
    try:
        os.remove(mdl_base+".index")
    except:
        pass

    try:
        os.remove(mdl_base+".meta")
    except:
        pass
    
    try:
        os.remove(output_model_dir + "online_metrics" + b + ".pkl")
    except:
        pass
    
    return



#############################################################################################################
# run one update
def run_one_update(train_op, CER, accuracy, loss_ctc, words, input_tensor_b, labels_b, filenames_b,
                   data, output_model_dir, sess, b, i, j, num_errors, saver, input_tensor, labels):
    if train_op is not None:
        pred = "train"
    else:
        pred = "pred"
    
    try:
        if train_op is not None:
            _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                                feed_dict={input_tensor: input_tensor_b, labels: labels_b})
        else:
            cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                             feed_dict={input_tensor: input_tensor_b, labels: labels_b})
        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                   "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
    except:
        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                   "labels":[[""]], "words":[[""]], "filenames":[[""]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print("Error at ", b, i, j)
        num_errors += 1
    # save data
    newdata = pd.DataFrame.from_dict(newdata)
    data = data.append(newdata)
    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")
    return data, num_errors



#############################################################################################################


# run a single bunch
def run_bunch(restore_model_nm, output_graph_dir, n_epochs_per_bunch, iterator,
              next_batch, n_batches, data, output_model_dir, train_op, 
              CER, accuracy, loss_ctc, words, do_predict, b, input_tensor, labels):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver.restore(sess, restore_model_nm)

        #writer = tf.summary.FileWriter(output_graph_dir, sess.graph)
        
        num_errors = 0 # monitor for errors and skip bunch when too many occur
        for i in range(n_epochs_per_bunch):
            sess.run(iterator.initializer)      
            print("---------------------------------------------------------")
            print("Starting epoch", i)
            for j in range(0, n_batches):
                input_tensor_b, labels_b, filenames_b = sess.run(next_batch)

                if i < 1 and do_predict: # only predict on first run through
                    data, num_errors = run_one_update(None, CER, accuracy, loss_ctc, words,
                                          input_tensor_b, labels_b, filenames_b,
                                          data, output_model_dir, sess, b, i, j,
                                          num_errors, saver, input_tensor, labels)
                    
                # train with new data
                data, num_errors = run_one_update(train_op, CER, accuracy, loss_ctc, words,
                                          input_tensor_b, labels_b, filenames_b,
                                          data, output_model_dir, sess, b, i, j,
                                          num_errors, saver, input_tensor, labels)
                
                if num_errors > n_batches/2.0: # if half the batch is errors, stop
                    print("Ending batch due to too many errors")
                    break
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
            if num_errors > n_epochs_per_bunch * n_batches/3.0: # if one third of the bunch is errors, stop
                print("Ending bunch due to too many errors")
                break
    
    return data

"""
#############################################################################################################
# run one update
def run_one_update(train_op, CER, accuracy, loss_ctc, words, input_tensor_b, labels_b, filenames_b,
                   data, output_model_dir, sess, b, i, j, num_errors, saver, input_tensor, labels):
    if train_op is not None:
        pred = "train"
    else:
        pred = "pred"
    
    try:
        print("here1")
        if train_op is not None:
            _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                                feed_dict={input_tensor: input_tensor_b, labels: labels_b})
        else:
            cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                             feed_dict={input_tensor: input_tensor_b, labels: labels_b})
        print("here2")
        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                   "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print("here3")
        print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
    except:
        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                   "labels":[[""]], "words":[[""]], "filenames":[[""]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print("Error at ", b, i, j)
        num_errors += 1
    # save data
    newdata = pd.DataFrame.from_dict(newdata)
    data = data.append(newdata)
    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")
    return data, num_errors



#############################################################################################################


# run a single bunch
def run_bunch(restore_model_nm, output_graph_dir, n_epochs_per_bunch, iterator,
              next_batch, n_batches, data, output_model_dir, train_op, 
              CER, accuracy, loss_ctc, words, do_predict, b, input_tensor, labels):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver.restore(sess, restore_model_nm)

        #writer = tf.summary.FileWriter(output_graph_dir, sess.graph)
        
        num_errors = 0 # monitor for errors and skip bunch when too many occur
        for i in range(n_epochs_per_bunch):
            sess.run(iterator.initializer)      
            print("---------------------------------------------------------")
            print("Starting epoch", i)
            for j in range(0, n_batches):
                input_tensor_b, labels_b, filenames_b = sess.run(next_batch)

                if i < 1 and do_predict: # only predict on first run through
                    data, num_errors = run_one_update(None, CER, accuracy, loss_ctc, words,
                                          input_tensor_b, labels_b, filenames_b,
                                          data, output_model_dir, sess, b, i, j,
                                          num_errors, saver, input_tensor, labels)
                    
                '''if i < 1 and do_predict: # only predict on first run through
                    pred = "pred"
                    #try:
                    cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                                     feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                               "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                               "pred":pred, "bunch":b, "epoch":i, "batch":j}
                    print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
                    #except:
                    #    newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                    #               "labels":[[""]], "words":[[""]], "filenames":[[""]],
                    #               "pred":pred, "bunch":b, "epoch":i, "batch":j}
                    #    print("Error at ", b, i, j)
                    #    num_errors += 1
                    # save data
                    newdata = pd.DataFrame.from_dict(newdata)
                    data = data.append(newdata)
                    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
                    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")'''
                    
                # train with new data
                pred = "train"
                try:
                    _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                                            feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                    newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                               "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                               "pred":pred, "bunch":b, "epoch":i, "batch":j}
                    print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
                except:
                    newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                               "labels":[[""]], "words":[[""]], "filenames":[[""]],
                               "pred":pred, "bunch":b, "epoch":i, "batch":j}
                    print("Error at ", b, i, j)
                    num_errors += 1
                # save data
                newdata = pd.DataFrame.from_dict(newdata)
                data = data.append(newdata)
                pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
                saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")
                
                if num_errors > n_batches/2.0: # if half the batch is errors, stop
                    print("Ending batch due to too many errors")
                    break
            print('Avg Epoch time: {0} seconds'.format((time.time() - start_time)/(1.0*(i+1))))
            if num_errors > n_epochs_per_bunch * n_batches/3.0: # if one third of the bunch is errors, stop
                print("Ending bunch due to too many errors")
                break
    
    return data



#############################################################################################################
# prediction
# do prediction first
                    pred = "pred"
                    try:
                        cer, acc, loss, wordz = sess.run([CER, accuracy, loss_ctc, words],
                                     feed_dict={input_tensor: input_tensor_b, labels: labels_b})
                        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                                  "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
                        print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
                    except:
                        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                                  "labels":[[""]], "words":[[""]], "filenames":[[""]],
                                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
                        print("Error at ", b, i, j)
                    # save data
                    newdata = pd.DataFrame.from_dict(newdata)
                    data = data.append(newdata)
                    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
                    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")

#############################################################################################################
# train prev
pred = "train"
    try:
        _, cer, acc, loss, wordz = sess.run([train_op, CER, accuracy, loss_ctc, words],
                                            feed_dict={input_tensor: input_tensor_b, labels: labels_b})
        newdata = {"loss":loss, "cer":cer, "accuracy":[[acc]], 
                   "labels":[[labels_b]], "words":[[wordz]], "filenames":[[filenames_b]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print('batch: {0}:{5}:{4}, loss: {3} \n\tCER: {1}, accuracy: {2}'.format(b, cer, acc, loss, j, i))
    except:
        newdata = {"loss":-1, "cer":-1, "accuracy":[[-1, -1]], 
                   "labels":[[""]], "words":[[""]], "filenames":[[""]],
                   "pred":pred, "bunch":b, "epoch":i, "batch":j}
        print("Error at ", b, i, j)
        num_errors += 1
    # save data
    newdata = pd.DataFrame.from_dict(newdata)
    data = data.append(newdata)
    pickle.dump(data, open(output_model_dir+"online_metrics" + str(b) + ".pkl", "wb"))
    saver.save(sess, output_model_dir+"online_model" + str(b) + ".ckpt")
    return data
"""
