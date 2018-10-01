import tensorflow as tf
from src.tools import generate_init_values


class FCLayer:
    def __init__(self, rnn_config, layer_idx):
        self.rnn_config = rnn_config
        self.layer_config = rnn_config['layer_configs'][layer_idx]
        self.w_shape = (rnn_config['layout'][layer_idx-1], rnn_config['layout'][layer_idx])
        self.b_shape = (1, self.w_shape[1])

        with tf.variable_scope(self.layer_config['var_scope']):
            w_init_vals, b_init_vals = generate_init_values(self.layer_config['init_config'], self.w_shape, self.b_shape)
            w_initializer = tf.constant_initializer(w_init_vals)
            b_initializer = tf.constant_initializer(b_init_vals)
            self.w = tf.get_variable(name='w', shape=self.w_shape, initializer=w_initializer)
            self.b = tf.get_variable(name='b', shape=self.b_shape, initializer=b_initializer)

    # Returns the output of the layer. If its the output layer, this only returns the activation!
    def create_forward_pass(self, layer_input):
        if self.layer_config['is_output']:
            return tf.matmul(layer_input, self.w) + self.b
        else:
            return self.layer_config['act_func'](tf.matmul(layer_input, self.w) + self.b)



