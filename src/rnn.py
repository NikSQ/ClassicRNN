import tensorflow as tf
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer
import copy


class RNN:
    def __init__(self, rnn_config, training_config, l_data):
        self.rnn_config = rnn_config
        self.l_data = l_data
        self.layers = []
        self.train_op = None
        self.accuracy = None
        self.gradients = None # Used to find a good value to clip

        self.metrics = dict()
        for key in self.l_data.data.keys():
            self.metrics[key] = dict()

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(rnn_config, training_config, layer_idx, self.is_training)
            elif layer_config['layer_type'] == 'lstm':
                layer = LSTMLayer(rnn_config, training_config, layer_idx, self.is_training)
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))

            self.layers.append(layer)

    def create_rnn_graph(self, data_key, mod_rnn_config):
        x_shape = self.l_data.data[data_key]['x_shape']
        y_shape = self.l_data.data[data_key]['y_shape']
        x = self.l_data.data[data_key]['x_batch']
        y = self.l_data.data[data_key]['y_batch']
        end = self.l_data.data[data_key]['end_batch']
        outputs = []

        for seq_idx in range(x_shape[2]):
            layer_input = x[:, :, seq_idx]
            for layer_idx, layer in enumerate(self.layers, 1):
                layer_input = layer.create_forward_pass(layer_input, mod_rnn_config['layer_configs'][layer_idx],
                                                        seq_idx)

            outputs.append(layer_input)
        output = tf.stack(outputs, axis=1)
        gather_idcs = tf.stack([tf.range(y_shape[0]), end], axis=1)
        output = tf.gather_nd(output, gather_idcs)
        reg_loss = 0
        for layer in self.layers:
            reg_loss += layer.layer_loss

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y, dim=1)) + reg_loss
        output = tf.nn.softmax(output, axis=1)
        prediction = tf.argmax(output, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)),
                                          dtype=tf.float32))
        self.metrics[data_key]['loss'] = loss
        self.metrics[data_key]['pred'] = prediction
        self.metrics[data_key]['acc'] = accuracy
        self.metrics[data_key]['out'] = output

    def create_training_graph(self):
        with tf.variable_scope('training'):
            self.create_rnn_graph('tr', self.rnn_config)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradients = optimizer.compute_gradients(self.metrics['tr']['loss'])

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in self.gradients]
            bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(bn_ops):
                self.train_op = optimizer.apply_gradients(clipped_gradients)

    def create_evaluation_graph(self, data_key):
        with tf.variable_scope(data_key):
            graph_config = copy.deepcopy(self.rnn_config)
            for layer_config in graph_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.create_rnn_graph(data_key, graph_config)
