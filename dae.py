from gaussianencoder import *
from decoder import *


class GaussianDenoisingAutoEncoder(object):
    def __init__(self, n_input, n_encoder_units, n_latent, n_decoder_units, n_output, lr, lam, scale, activation):
        self._n_input = n_input
        self._n_encoder_units = n_encoder_units
        self._n_latent = n_latent
        self._n_decoder_units = n_decoder_units
        self._n_output = n_output
        self._activation = activation

        self._lr = lr
        self._lam = lam
        self._scale = scale

        self._x_pl = tf.placeholder(tf.float32, shape=[None, n_input], name='x_pl')
        self._enc = GaussianEncoder(n_input, n_encoder_units, n_latent, scale, activation)
        self._dec = Decoder(n_latent, n_decoder_units, n_output, activation)

        self._latent = self._enc.forward(self._x_pl)
        self._reconstruct = self._dec.forward(self._latent)

        self._reconstruct_loss = tf.reduce_mean(tf.square(self._reconstruct - self._x_pl))
        self._wd_loss = lam * (self._enc.wd_loss + self._dec.wd_loss)
        self._loss = self._reconstruct_loss + self._wd_loss

        global_step = tf.Variable(0, name="global_step", trainable=False)
        t_vars = tf.trainable_variables()
        grads = tf.gradients(ys=self._loss, xs=t_vars)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, name="ADAM_optimizer")

        self._train_op = self._optimizer.apply_gradients(grads_and_vars=zip(grads, t_vars),
                                                         global_step=global_step,
                                                         name="train_op")

    @property
    def x_pl(self):
        return self._x_pl

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def reconstruct(self):
        return self._reconstruct

    @property
    def latent(self):
        return self._latent


class MaskingDenoisingAutoEncoder(object):
    def __init__(self, n_input, n_encoder_units, n_latent, n_decoder_units, n_output, lr, lam, keep_prob, activation):
        self._n_input = n_input
        self._n_encoder_units = n_encoder_units
        self._n_latent = n_latent
        self._n_decoder_units = n_decoder_units
        self._n_output = n_output
        self._activation = activation

        self._lr = lr
        self._lam = lam
        self._keep_prob = keep_prob

        self._x_pl = tf.placeholder(tf.float32, shape=[None, n_input], name='x_pl')
        self._enc = GaussianEncoder(n_input, n_encoder_units, n_latent, keep_prob, activation)
        self._dec = Decoder(n_latent, n_decoder_units, n_output, activation)

        self._latent = self._enc.forward(self._x_pl)
        self._reconstruct = self._dec.forward(self._latent)

        self._reconstruct_loss = tf.reduce_mean(tf.square(self._reconstruct - self._x_pl))
        self._wd_loss = lam * (self._enc.wd_loss + self._dec.wd_loss)
        self._loss = self._reconstruct_loss + self._wd_loss

        global_step = tf.Variable(0, name="global_step", trainable=False)
        t_vars = tf.trainable_variables()
        grads = tf.gradients(ys=self._loss, xs=t_vars)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, name="ADAM_optimizer")

        self._train_op = self._optimizer.apply_gradients(grads_and_vars=zip(grads, t_vars),
                                                         global_step=global_step,
                                                         name="train_op")

    @property
    def x_pl(self):
        return self._x_pl

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def reconstruct(self):
        return self._reconstruct

    @property
    def latent(self):
        return self._latent
