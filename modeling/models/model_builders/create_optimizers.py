import tensorflow as tf

def create_optimizer(loss, optimizer="adam", learning_rate=1e-3, 
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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    opt_op = optimizer.minimize(loss, global_step=global_step)
    with tf.control_dependencies(update_ops + [opt_op]):
        train_op = tf.group(maintain_averages_op)

    return train_op