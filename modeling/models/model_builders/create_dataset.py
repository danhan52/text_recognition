import tensorflow as tf

def create_iterator(csv_files_train, input_shape, batch_size, shuffle=True):
    with open(csv_files_train, "r") as f:
        f_lines = f.read().splitlines()[1:]
        filenames = [l.split("\t")[0] for l in f_lines]
        label_list = [l.split("\t")[1] for l in f_lines]

    datasize = len(label_list)
    filenames = tf.constant(filenames)
    label_list = tf.constant(label_list)
        
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded,
                                                               input_shape[0],
                                                               input_shape[1])
        return image_resized, label
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, label_list))
    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=datasize)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    return dataset, iterator, next_batch, datasize