from layers import *


class GaussianEncoder(object):

    def __init__(self, n_input, n_units, n_latent, scale, activation):
        self._n_input = n_input
        self._n_units = n_units
        self._n_latent = n_latent
        self._activation = activation
        self._scale = scale

        self._n_layer = len(n_units)
        self._hidden_layers = list()
        for layer_idx in range(self._n_layer):
            layer_name = "encoder_layer_" + str(layer_idx + 1)
            if layer_idx is 0:
                n_layer_input = n_input
            else:
                n_layer_input = n_units[layer_idx - 1]
            n_unit = n_units[layer_idx]
            self._hidden_layers.append(
                AffinePlusNonlinearLayerWithGaussianNoise(layer_name, n_layer_input, n_unit, scale, activation))

        layer_name = "latent_layer"
        self._latent_layer = AffinePlusNonlinearLayerWithGaussianNoise(
            layer_name, n_units[-1], n_latent, scale, activation)

        self._weight_decay_loss = self._latent_layer.get_weight_decay_loss()
        for layer_idx in range(self._n_layer):
            self._weight_decay_loss += self._hidden_layers[layer_idx].get_weight_decay_loss()

    def forward(self, input_tensor):
        output_tensor = input_tensor
        for layer_idx in range(self._n_layer):
            output_tensor = self._hidden_layers[layer_idx].forward(output_tensor)
        output_tensor = self._latent_layer.forward(output_tensor)
        return output_tensor

    @property
    def wd_loss(self):
        return self._weight_decay_loss


