import tensorflow as tf
from src.tools import generate_init_values
from src.tools import get_batchnormalizer
import numpy as np


class FCLayer:
    def __init__(self, rnn_config, train_config, layer_idx, is_training):
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.is_training = is_training
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        if 'fc' in self.train_config['batchnorm']['modes']:
            self.bn_x = get_batchnormalizer()

        with tf.variable_scope(self.layer_config['var_scope']):
            w_init_vals, b_init_vals = generate_init_values(self.layer_config['init_config'], self.w_shape, self.b_shape)
            w_initializer = tf.constant_initializer(w_init_vals)
            b_initializer = tf.constant_initializer(b_init_vals)
            self.w = tf.get_variable(name='w', shape=self.w_shape, initializer=w_initializer)
            self.b = tf.get_variable(name='b', shape=self.b_shape, initializer=b_initializer)

            if self.layer_config['regularization']['mode'] == 'l2':
                self.layer_loss = self.layer_config['regularization']['strength'] * tf.nn.l2_loss(self.w)
            else:
                self.layer_loss = 0

    # Returns the output of the layer. If its the output layer, this only returns the activation!
    def create_forward_pass(self, x, mod_layer_config, init, time_idx):
        if self.train_config['batchnorm']['type'] == 'batch' and 'fc' in self.train_config['batchnorm']:
            x = self.bn_x(x, self.is_training)
        act = tf.matmul(x, self.w) + self.b
        if self.train_config['batchnorm']['type'] != 'layer':
            return tf.contrib.layers.layer_norm(act)
        else:
            return act



