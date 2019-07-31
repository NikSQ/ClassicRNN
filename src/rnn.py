import tensorflow as tf
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer
import copy


class RNN:
    def __init__(self, rnn_config, labelled_data):
        self.rnn_config = rnn_config
        self.labelled_data = labelled_data
        self.layers = []
        self.train_op = None
        self.accuracy = None
        self.gradients = None # Used to find a good value to clip

        self.tr_acc = None
        self.tr_pred = None
        self.tr_loss = None
        self.tr_out = None

        self.va_acc = None
        self.va_pred = None
        self.va_loss = None
        self.va_out = None

        self.te_acc = None
        self.te_pred = None
        self.te_loss = None
        self.te_out = None

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(rnn_config, layer_idx, self.is_training)
            elif layer_config['layer_type'] == 'lstm':
                layer = LSTMLayer(rnn_config, layer_idx, self.is_training)
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))

            self.layers.append(layer)

    def create_rnn_graph(self, x, y, end_times, x_shape, y_shape, mod_rnn_config):
        outputs = []

        for seq_idx in range(x_shape[2]):
            layer_input = x[:, :, seq_idx]
            for layer_idx, layer in enumerate(self.layers, 1):
                layer_input = layer.create_forward_pass(layer_input, mod_rnn_config['layer_configs'][layer_idx],
                                                        seq_idx == 0)

            outputs.append(layer_input)

        output = tf.stack(outputs, axis=1)
        gather_idcs = tf.stack([tf.range(y_shape[0]), end_times], axis=1)
        output = tf.gather_nd(output, gather_idcs)
        reg_loss = 0
        for layer in self.layers:
            reg_loss += layer.layer_loss

        if self.rnn_config['output_type'] == 'regression':
            loss = tf.reduce_mean(tf.square(output - y)) + reg_loss
            prediction = output
            accuracy = None
            output = None
        elif self.rnn_config['output_type'] == 'classification':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y, dim=1)) + reg_loss
            output = tf.nn.softmax(output, axis=1)
            prediction = tf.argmax(output, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)),
                                              dtype=tf.float32))

        return loss, prediction, accuracy, output

    def create_training_graph(self):
        with tf.variable_scope('training'):
            self.tr_loss, self.tr_pred, self.tr_acc, self.tr_out = \
                self.create_rnn_graph(self.labelled_data.x_tr_batch, self.labelled_data.y_tr_batch, self.labelled_data.end_tr_batch,
                                      self.labelled_data.x_tr_shape, self.labelled_data.y_tr_shape, self.rnn_config)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.gradients = optimizer.compute_gradients(self.tr_loss)

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var)
                                 for grad, var in self.gradients]
            self.train_op = optimizer.apply_gradients(clipped_gradients)

    def create_validation_graph(self):
        with tf.variable_scope('validation'):
            graph_config = copy.deepcopy(self.rnn_config)
            for layer_config in graph_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.va_loss, self.va_pred, self.va_acc, self.va_out = \
                self.create_rnn_graph(self.labelled_data.x_va_batch, self.labelled_data.y_va_batch, self.labelled_data.end_va_batch,
                                      self.labelled_data.x_va_shape, self.labelled_data.y_va_shape, graph_config)

    def create_test_graph(self):
        with tf.variable_scope('test'):
            graph_config = copy.deepcopy(self.rnn_config)
            for layer_config in graph_config['layer_configs']:
                if 'regularization' in layer_config:
                    layer_config['regularization']['mode'] = None

            self.te_loss, self.te_pred, self.te_acc, self.te_out = \
                self.create_rnn_graph(self.labelled_data.x_te_batch, self.labelled_data.y_te_batch, self.labelled_data.end_te_batch,
                                      self.labelled_data.x_te_shape, self.labelled_data.y_te_shape, graph_config)
