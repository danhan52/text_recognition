import tensorflow as tf

def create_optimizer(loss, lastlayer=False, optimizer="adam", learning_rate=1e-3, 
    learning_decay_rate=0.95, learning_decay_steps=5000):
    # Create the learning rate as well as a moving average
    global_step = tf.train.get_or_create_global_step()
    # # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step, zero_debias=True)
    # Create the shadow variables, and add op to maintain moving averages
    maintain_averages_op = ema.apply([loss])
    loss_ema = ema.average(loss)

    # Train op
    # --------
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               learning_decay_steps, learning_decay_rate,
                                               staircase=True)

    # Set up optimizer
    if optimizer == 'ada':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    elif optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

    if lastlayer:
        connected_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fully_connected")
        opt_op = optimizer.minimize(loss, global_step=global_step, var_list=connected_vars)
    else:
        opt_op = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops + [opt_op]):
        train_op = tf.group(maintain_averages_op)

    return train_op