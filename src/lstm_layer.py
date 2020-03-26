import tensorflow as tf
import numpy as np
from src.tools import generate_init_values
from src.tools import get_batchnormalizer


@tf.custom_gradient
def ternarize_weight(w):
    w = tf.cast(tf.cast(w + 0.5, dtype=tf.int32), tf.float32)
    w = tf.clip_by_value(w, -1., 1.)
    def grad(dy):
        return dy
    return w, grad


@tf.custom_gradient
def binarize_weight(w):
    w = tf.cast(tf.greater_equal(w, 0.), tf.float32) * 2. - 1.
    def grad(dy):
        return dy
    return w, grad


@tf.custom_gradient
def disc_sigmoid(act, n_bins):
    s_act = tf.sigmoid(act)
    disc_output = tf.cast(tf.cast(s_act*n_bins, dtype=tf.int32), dtype=tf.float32) / n_bins
    def grad(dy):
        return dy * tf.multiply(s_act, 1-s_act), tf.zeros_like(n_bins)
    return disc_output, grad


@tf.custom_gradient
def disc_tanh(act, n_bins):
    disc_output = tf.cast(tf.cast(tf.sigmoid(act) * n_bins, dtype=tf.int32), dtype=tf.float32) * 2 / n_bins - 1
    def grad(dy):
        return dy * (1 - tf.square(tf.tanh(act))), tf.zeros_like(n_bins)
    return disc_output, grad

class LSTMLayer:
    def __init__(self, rnn_config, train_config, layer_idx, is_training, prev_blstm=False):
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.is_training = is_training
        self.layer_idx = layer_idx
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        if prev_blstm:
            self.w_shape = (self.rnn_config['layout'][layer_idx-1]*2 + self.rnn_config['layout'][layer_idx],
                            self.rnn_config['layout'][layer_idx])
        else:
            self.w_shape = (self.rnn_config['layout'][layer_idx-1] + self.rnn_config['layout'][layer_idx],
                            self.rnn_config['layout'][layer_idx])

        self.b_shape = (1, self.w_shape[1])

        if 'x' in self.train_config['batchnorm']['modes']:
            self.bn_x = []
        if 'h' in self.train_config['batchnorm']['modes']:
            self.bn_h = []

        with tf.variable_scope(self.layer_config['var_scope']):
            wi_init_vals, bi_init_vals = generate_init_values(self.layer_config['init_config']['i'],
                                                              self.w_shape, self.b_shape)
            wc_init_vals, bc_init_vals = generate_init_values(self.layer_config['init_config']['c'],
                                                              self.w_shape, self.b_shape)
            wo_init_vals, bo_init_vals = generate_init_values(self.layer_config['init_config']['o'],
                                                              self.w_shape, self.b_shape)
            wi_initializer = tf.constant_initializer(wi_init_vals)
            bi_initializer = tf.constant_initializer(bi_init_vals)
            wc_initializer = tf.constant_initializer(wc_init_vals)
            bc_initializer = tf.constant_initializer(bc_init_vals)
            wo_initializer = tf.constant_initializer(wo_init_vals)
            bo_initializer = tf.constant_initializer(bo_init_vals)

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

    def create_forward_pass(self, layer_input, mod_layer_config, init, time_idx):
        if init:
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

        if self.rnn_config['weight_type'] == 'continuous':
            i_act = self.bi + tf.matmul(x, self.wi, name='i')
            o_act = self.bo + tf.matmul(x, self.wo, name='o')
            c_act = self.bc + tf.matmul(x, self.wc, name='c')
        elif self.rnn_config['weight_type'] == 'ternary':
            i_act = self.bi + tf.matmul(x, ternarize_weight(self.wi), name='i')
            o_act = self.bo + tf.matmul(x, ternarize_weight(self.wo), name='o')
            c_act = self.bc + tf.matmul(x, ternarize_weight(self.wc), name='c')
        elif self.rnn_config['weight_type'] == 'binary':
            i_act = self.bi + tf.matmul(x, binarize_weight(self.wi), name='i')
            o_act = self.bo + tf.matmul(x, binarize_weight(self.wo), name='o')
            c_act = self.bc + tf.matmul(x, binarize_weight(self.wc), name='c')
        else:
            raise Exception('weight type not understood')

        if self.train_config['batchnorm']['type'] == 'layer':
            i_act = tf.contrib.layers.layer_norm(i_act)
            o_act = tf.contrib.layers.layer_norm(o_act)
            c_act = tf.contrib.layers.layer_norm(c_act)


        i = tf.sigmoid(i_act)
        o = tf.sigmoid(o_act)
        c = tf.tanh(c_act)
        if 'i' in self.rnn_config['discrete_acts']:
            i = disc_sigmoid(i_act, self.rnn_config['n_bins'])
        if 'c' in self.rnn_config['discrete_acts']:
            c = disc_tanh(c_act, self.rnn_config['n_bins'])
        if 'o' in self.rnn_config['discrete_acts']:
            o = disc_sigmoid(o_act, self.rnn_config['n_bins'])

        f = 1. - i

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

