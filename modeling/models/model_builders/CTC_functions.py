import tensorflow as tf
from typing import List

def get_words_from_chars(characters_list: List[str], sequence_lengths: List[int], name='chars_conversion'):
    with tf.name_scope(name=name):
        def join_characters_fn(coords):
            return tf.reduce_join(characters_list[coords[0]:coords[1]])

        def coords_several_sequences():
            end_coords = tf.cumsum(sequence_lengths)
            start_coords = tf.concat([[0], end_coords[:-1]], axis=0)
            coords = tf.stack([start_coords, end_coords], axis=1)
            coords = tf.cast(coords, dtype=tf.int32)
            return tf.map_fn(join_characters_fn, coords, dtype=tf.string)

        def coords_single_sequence():
            return tf.reduce_join(characters_list, keep_dims=True)

        words = tf.cond(tf.shape(sequence_lengths)[0] > 1,
                        true_fn=lambda: coords_several_sequences(),
                        false_fn=lambda: coords_single_sequence())

    return words

def ctc_loss(prob, labels, input_shape, alphabet, alphabet_codes, batch_size, 
    n_pools=2*2, decode=True):
    # Compute seq_len from image width
    # 2x2 pooling in dimension W on layer 1 and 2 -> n-pools = 2*2
    seq_len_inputs = tf.divide([input_shape[1]]*batch_size, n_pools,
                               name='seq_len_input_op') - 1

    # Get keys (letters) and values (integer stand ins for letters)
    # Alphabet and codes
    keys = [c for c in alphabet] # the letters themselves
    values = alphabet_codes # integer representations


    # Create non-string labels from the keys and values above
    # Convert string label to code label
    with tf.name_scope('str2code_conversion'):
        table_str2int = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
        splited = tf.string_split(labels, delimiter='')  # TODO change string split to utf8 split in next tf version
        codes = table_str2int.lookup(splited.values)
        sparse_code_target = tf.SparseTensor(splited.indices, codes, splited.dense_shape)

    seq_lengths_labels = tf.bincount(tf.cast(sparse_code_target.indices[:, 0], tf.int32),
                                     minlength=tf.shape(prob)[1])


    # Use ctc loss on probabilities from lstm output
    # Loss
    # ----
    # >>> Cannot have longer labels than predictions -> error
    with tf.control_dependencies([tf.less_equal(sparse_code_target.dense_shape[1], tf.reduce_max(tf.cast(seq_len_inputs, tf.int64)))]):
        loss_ctc = tf.nn.ctc_loss(labels=sparse_code_target,
                                  inputs=prob,
                                  sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                  preprocess_collapse_repeated=False,
                                  ctc_merge_repeated=True,
                                  ignore_longer_outputs_than_inputs=True,  # returns zero gradient in case it happens -> ema loss = NaN
                                  time_major=True)
        loss_ctc = tf.reduce_mean(loss_ctc)
        loss_ctc = tf.Print(loss_ctc, [loss_ctc], message='* Loss : ')

    if decode:
        with tf.name_scope('code2str_conversion'):
            keys = tf.cast(alphabet_codes, tf.int64)
            values = [c for c in alphabet]

            table_int2str = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '?')

            sparse_code_pred, log_probability = tf.nn.ctc_beam_search_decoder(prob,
                                                                              sequence_length=tf.cast(seq_len_inputs, tf.int32),
                                                                              merge_repeated=False,
                                                                              beam_width=100,
                                                                              top_paths=2)
            # Score
            pred_score = tf.subtract(log_probability[:, 0], log_probability[:, 1])

            sparse_code_pred = sparse_code_pred[0]

            sequence_lengths_pred = tf.bincount(tf.cast(sparse_code_pred.indices[:, 0], tf.int32),
                                                minlength=tf.shape(prob)[1])

            pred_chars = table_int2str.lookup(sparse_code_pred)
            words = get_words_from_chars(pred_chars.values, sequence_lengths=sequence_lengths_pred)

            # tf.summary.text('predicted_words', words[:10])

        with tf.name_scope('evaluation'):
            CER = tf.metrics.mean(tf.edit_distance(sparse_code_pred, tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')
            CER = tf.reduce_mean(tf.edit_distance(sparse_code_pred, tf.cast(sparse_code_target, dtype=tf.int64)), name='CER')

            # Convert label codes to decoding alphabet to compare predicted and groundtrouth words
            target_chars = table_int2str.lookup(tf.cast(sparse_code_target, tf.int64))
            target_words = get_words_from_chars(target_chars.values, seq_lengths_labels)
            accuracy = tf.metrics.accuracy(target_words, words, name='accuracy')

            CER = tf.Print(CER, [CER], message='-- CER : ')
            accuracy = tf.Print(accuracy, [accuracy], message='-- Accuracy : ')
    else:
        CER = None; accuracy = None


    return loss_ctc, words, pred_score, CER, accuracy
