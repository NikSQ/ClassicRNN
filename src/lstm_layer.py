import tensorflow as tf
import numpy as np
from src.tools import generate_init_values
from src.tools import get_batchnormalizer

class LSTMLayer:
    def __init__(self, rnn_config, train_config, layer_idx, is_training):
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.is_training = is_training
        self.layer_idx = layer_idx
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1] + rnn_config['layout'][layer_idx],
                   rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        if 'x' in self.train_config['batchnorm']['modes']:
            self.bn_x = []
        if 'h' in self.train_config['batchnorm']['modes']:
            self.bn_h = []

        with tf.variable_scope(self.layer_config['var_scope']):
            wf_init_vals, bf_init_vals = generate_init_values(self.layer_config['init_config']['f'],
                                                              self.w_shape, self.b_shape)
            wi_init_vals, bi_init_vals = generate_init_values(self.layer_config['init_config']['i'],
                                                              self.w_shape, self.b_shape)
            wc_init_vals, bc_init_vals = generate_init_values(self.layer_config['init_config']['c'],
                                                              self.w_shape, self.b_shape)
            wo_init_vals, bo_init_vals = generate_init_values(self.layer_config['init_config']['o'],
                                                              self.w_shape, self.b_shape)
            wf_initializer = tf.constant_initializer(wf_init_vals)
            bf_initializer = tf.constant_initializer(bf_init_vals)
            wi_initializer = tf.constant_initializer(wi_init_vals)
            bi_initializer = tf.constant_initializer(bi_init_vals)
            wc_initializer = tf.constant_initializer(wc_init_vals)
            bc_initializer = tf.constant_initializer(bc_init_vals)
            wo_initializer = tf.constant_initializer(wo_init_vals)
            bo_initializer = tf.constant_initializer(bo_init_vals)

            self.wf = tf.get_variable(name='wf', shape=self.w_shape, initializer=wf_initializer)
            self.bf = tf.get_variable(name='bf', shape=self.b_shape, initializer=bf_initializer)
            self.wi = tf.get_variable(name='wi', shape=self.w_shape, initializer=wi_initializer)
            self.bi = tf.get_variable(name='bi', shape=self.b_shape, initializer=bi_initializer)
            self.wc = tf.get_variable(name='wc', shape=self.w_shape, initializer=wc_initializer)
            self.bc = tf.get_variable(name='bc', shape=self.b_shape, initializer=bc_initializer)
            self.wo = tf.get_variable(name='wo', shape=self.w_shape, initializer=wo_initializer)
            self.bo = tf.get_variable(name='bo', shape=self.b_shape, initializer=bo_initializer)

            self.cell_state = None
            self.cell_output = None

            if self.layer_config['regularization']['mode'] == 'zoneout':
                self.state_zoneout_dist = tf.distributions.Bernoulli(probs=
                                                                     self.layer_config['regularization']['state_zo_prob'],
                                                                     dtype=tf.float32, allow_nan_stats=False)
                self.output_zoneout_dist = tf.distributions.Bernoulli(probs=
                                                                      self.layer_config['regularization']['output_zo_prob'],
                                                                      dtype=tf.float32, allow_nan_stats=False)
            self.layer_loss = 0

    def create_forward_pass(self, layer_input, mod_layer_config, time_idx):
        if time_idx == 0:
            cell_shape = (tf.shape(layer_input)[0], self.b_shape[1])
            self.cell_state = tf.zeros(cell_shape)
            self.cell_output = tf.zeros(cell_shape)

        co = self.cell_output
        if self.train_config['batchnorm']['type'] == 'batch':
            bn_idx = min(time_idx, self.train_config['batchnorm']['tau'] - 1)
            if 'x' in self.train_config['batchnorm']['modes']:
                if len(self.bn_x) == bn_idx:
                    self.bn_x.append(get_batchnormalizer())
                layer_input = self.bn_x[bn_idx](layer_input, self.is_training)
            if 'h' in self.train_config['batchnorm']['modes'] and bn_idx > 0:
                if len(self.bn_h) == bn_idx - 1:
                    self.bn_h.append(get_batchnormalizer())
                co = self.bn_h[bn_idx - 1](self.cell_output, self.is_training)

        x = tf.concat([layer_input, co], axis=1)
        if self.train_config['batchnorm']['type'] == 'layer':
            x = tf.contrib.layers.layer_norm(x)

        i = tf.sigmoid(self.bi + tf.matmul(x, self.wi, name='i'))
        f = 1. - i
        o = tf.sigmoid(self.bo + tf.matmul(x, self.wo, name='o'))
        c = tf.tanh(self.bc + tf.matmul(x, self.wc, name='c'))

        updated_state = tf.multiply(f, self.cell_state, name='f_cs') + tf.multiply(i, c, name='i_cs')
        updated_output = tf.multiply(o, tf.tanh(updated_state))

        if mod_layer_config['regularization']['mode'] == 'zoneout':
            state_mask = self.state_zoneout_dist.sample(tf.shape(self.cell_state))
            output_mask = self.output_zoneout_dist.sample(tf.shape(self.cell_output))
            self.cell_state = tf.multiply(state_mask, self.cell_state) + tf.multiply(1 - state_mask, updated_state)
            self.cell_output = tf.multiply(output_mask, self.cell_output) + tf.multiply(1 - output_mask, updated_output)
        else:
            self.cell_state = updated_state
            self.cell_output = updated_output

        return self.cell_output

