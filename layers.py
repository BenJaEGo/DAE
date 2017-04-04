from tf_tools import *
from ops import *


class AffinePlusNonlinearLayer(object):
    def __init__(self, name, n_input, n_hidden, activation=None):
        self._name = name
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._activation = activation

        with tf.variable_scope(name):
            self._weights = get_weights_normal(name="weights", shape=[n_input, n_hidden])
            self._biases = get_biases(name="biases", shape=[n_hidden])

    def forward(self, input_tensor):
        output_tensor = tf.matmul(input_tensor, self._weights) + self._biases
        if self._activation is not None:
            output_tensor = self._activation(output_tensor)
        return output_tensor

    def get_weight_decay_loss(self):
        return tf.nn.l2_loss(t=self._weights)


class AffinePlusNonlinearLayerWithGaussianNoise(object):
    def __init__(self, name, n_input, n_hidden, scale, activation=None):
        self._name = name
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._activation = activation
        self._scale = scale

        with tf.variable_scope(name):
            self._weights = get_weights_normal(name="weights", shape=[n_input, n_hidden])
            self._biases = get_biases(name="biases", shape=[n_hidden])

    def forward(self, input_tensor):
        input_tensor = input_tensor + self._scale * tf.random_normal((self._n_input,))
        output_tensor = tf.matmul(input_tensor, self._weights) + self._biases
        if self._activation is not None:
            output_tensor = self._activation(output_tensor)
        return output_tensor

    def get_weight_decay_loss(self):
        return tf.nn.l2_loss(t=self._weights)


class AffinePlusNonlinearLayerWithDropout(object):
    def __init__(self, name, n_input, n_hidden, keep_prob, activation=None):
        self._name = name
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._activation = activation
        self._keep_prob = keep_prob

        with tf.variable_scope(name):
            self._weights = get_weights_normal(name="weights", shape=[n_input, n_hidden])
            self._biases = get_biases(name="biases", shape=[n_hidden])

    def forward(self, input_tensor):
        input_tensor = tf.nn.dropout(input_tensor, keep_prob=self._keep_prob)
        output_tensor = tf.matmul(input_tensor, self._weights) + self._biases
        if self._activation is not None:
            output_tensor = self._activation(output_tensor)
        return output_tensor

    def get_weight_decay_loss(self):
        return tf.nn.l2_loss(t=self._weights)
