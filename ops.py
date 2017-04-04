import tensorflow as tf


def get_weights_normal(name, shape, mean=0.0, stddev=0.1):
    variable = tf.get_variable(name=name, shape=shape,
                               initializer=tf.truncated_normal_initializer(
                                   mean=mean,
                                   stddev=stddev
                               ))
    return variable


def get_biases(name, shape, init_value=0.0):
    variable = tf.get_variable(name=name, shape=shape,
                               initializer=tf.constant_initializer(value=init_value))
    return variable


def get_weights_uniform(name, shape, init_scale=0.1):
    variable = tf.get_variable(name=name, shape=shape,
                               initializer=tf.random_uniform_initializer(-init_scale, init_scale))
    return variable