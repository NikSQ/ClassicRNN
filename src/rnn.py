import tensorflow as tf
from src.fc_layer import FCLayer
from src.lstm_layer import LSTMLayer


class RNN:
    def __init__(self, rnn_config):
        self.rnn_config = rnn_config
        self.layers = []
        self.train_op = None
        self.labelled_data = None

        with tf.variable_scope('global'):
            self.learning_rate = tf.placeholder(tf.float32)

        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(rnn_config, layer_idx)
            elif layer_config['layer_type'] == 'lstm':
                layer = LSTMLayer(rnn_config, layer_idx)
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))

            self.layers.append(layer)

    def create_training_graph(self, labelled_data):
        self.labelled_data = labelled_data
        with tf.variable_scope('training'):
            outputs = []
            start_output_idx = labelled_data.x_shape[1] - labelled_data.y_shape[1]

            for seq_idx in range(labelled_data.x_shape[1]):
                layer_input = labelled_data.x[:, :, seq_idx]
                for layer in self.layers:
                    if (seq_idx >= start_output_idx) or (layer.layer_config['is_recurrent'] is True):
                        layer_input = layer.create_forward_pass(layer_input)
                if seq_idx >= start_output_idx:
                    outputs.append(layer_input)

            output = tf.concat(outputs, axis=0)

            if self.rnn_config['output_type'] == 'regression':
                loss = tf.reduce_mean(tf.square(output - labelled_data.y))
            elif self.rnn_config['output_type'] == 'classification':
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labelled_data.y))

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = optimizer.compute_gradients(loss)

            clipped_gradients = [(grad, var) if grad is None else
                                 (tf.clip_by_value(grad, -self.rnn_config['gradient_clip_value'],
                                                   self.rnn_config['gradient_clip_value']), var) for grad, var in gradients]
            self.train_op = optimizer.apply_gradients(clipped_gradients)









