import tensorflow as tf
from src.tools import generate_init_values

class LSTMLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1] + rnn_config['layout'][layer_idx],
                   rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

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

    def create_forward_pass(self, layer_input, mod_layer_config, do_initialize):
        if do_initialize:
            cell_shape = (tf.shape(layer_input)[0], self.b_shape[1])
            self.cell_state = tf.zeros(cell_shape)
            self.cell_output = tf.zeros(cell_shape)

        layer_input = tf.concat([layer_input, self.cell_output], axis=1)
        f = tf.sigmoid(tf.matmul(layer_input, self.wf) + self.bf)
        i = tf.sigmoid(tf.matmul(layer_input, self.wi) + self.bi)
        o = tf.sigmoid(tf.matmul(layer_input, self.wo) + self.bo)
        c = tf.tanh(tf.matmul(layer_input, self.wc) + self.bc)

        updated_state = tf.multiply(f, self.cell_state) + tf.multiply(i, c)
        updated_output = tf.multiply(o, tf.tanh(self.cell_state))

        if mod_layer_config['regularization']['mode'] == 'zoneout':
            state_mask = self.state_zoneout_dist.sample(tf.shape(self.cell_state))
            output_mask = self.output_zoneout_dist.sample(tf.shape(self.cell_output))
            self.cell_state = tf.multiply(state_mask, self.cell_state) + tf.multiply(1 - state_mask, updated_state)
            self.cell_output = tf.multiply(output_mask, self.cell_output) + tf.multiply(1 - output_mask, updated_output)
        else:
            self.cell_state = updated_state
            self.cell_output = updated_output

        return self.cell_output
