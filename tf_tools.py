import numpy as np
import tensorflow as tf


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def fill_feed_dict(x, y, batch_size, n_units, model):
    feed_dict = dict()
    feed_dict[model.x_pl] = x
    feed_dict[model.y_pl] = y
    n_layer = len(n_units)
    for i in range(n_layer):
        n_unit = n_units[i]
        feed_dict[model.rnn_initial_states[i]] = np.zeros(shape=[batch_size, n_unit])
        feed_dict[model.rnn_initial_outputs[i]] = np.zeros(shape=[batch_size, n_unit])

    return feed_dict