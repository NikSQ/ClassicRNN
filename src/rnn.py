import tensorflow as tf
import numpy as np
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

        bilstm = False
        for layer_idx, layer_config in enumerate(self.rnn_config['layer_configs']):
            if layer_config['layer_type'] == 'fc':
                layer = FCLayer(rnn_config, training_config, layer_idx, self.is_training)
            elif layer_config['layer_type'] == 'lstm' or layer_config['layer_type'] == 'blstm':
                layer = LSTMLayer(rnn_config, training_config, layer_idx, self.is_training, bilstm)
                if layer_config['layer_type'] == 'blstm':
                    bilstm = True
                else:
                    bilstm = False
            elif layer_config['layer_type'] == 'input':
                continue
            else:
                raise Exception("{} is not a valid layer type".format(layer_config['layer_type']))

            self.layers.append(layer)

    def unfold_rnn_layer(self, layer, layer_idx, layer_input, x_shape, mod_rnn_config, reverse=False):
        if reverse:
            loop_range = np.arange(x_shape[2])[::-1]
        else:
            loop_range = np.arange(x_shape[2])
        layer_output = []
        for time_idx in loop_range:
            m = layer_input[time_idx]
            init = time_idx == list(loop_range)[0]
            m = layer.create_forward_pass(m, mod_rnn_config['layer_configs'][layer_idx], init, time_idx)
            layer_output.append(m)
        return layer_output

    def unfold_rnn(self, x, x_shape, mod_rnn_config):
        for layer_idx, layer in enumerate(self.layers, 1):
            if layer_idx == 1:
                layer_input = tf.transpose(x, perm=[2,0,1])
            if layer.layer_config['layer_type'] == 'blstm':
                layer_input1 = self.unfold_rnn_layer(layer, layer_idx, layer_input, x_shape,
                                                     mod_rnn_config, reverse=False)
                layer_input2 = self.unfold_rnn_layer(layer, layer_idx, layer_input, x_shape,
                                                     mod_rnn_config, reverse=True)
                layer_input = []
                for input1, input2 in zip(layer_input1, layer_input2):
                    layer_input.append(tf.concat([input1, input2], axis=1))
            else:
                layer_input = self.unfold_rnn_layer(layer, layer_idx, layer_input, x_shape,
                                                    mod_rnn_config, reverse=False)
        return layer_input

    def create_rnn_graph(self, data_key, mod_rnn_config):
        x_shape = self.l_data.data[data_key]['x_shape']
        y_shape = self.l_data.data[data_key]['y_shape']
        x = self.l_data.data[data_key]['x_batch']
        y = self.l_data.data[data_key]['y_batch']
        end = self.l_data.data[data_key]['end_batch']

        output = self.unfold_rnn(x, x_shape, mod_rnn_config)
        output = tf.stack(output, axis=-1)
        one_hot = tf.one_hot(end, x_shape[2])[:, tf.newaxis, :]
        output = tf.multiply(output, one_hot)
        output = tf.reduce_sum(output, axis=2)

        #for seq_idx in range(x_shape[2]):
            #layer_input = x[:, :, seq_idx]
            #for layer_idx, layer in enumerate(self.layers, 1):
                #layer_input = layer.create_forward_pass(layer_input, mod_rnn_config['layer_configs'][layer_idx],
                                                        #seq_idx)
#
            #outputs.append(layer_input)
        #output = outputs[-1]
        #output = tf.stack(outputs, axis=1)
        #gather_idcs = tf.stack([tf.range(y_shape[0]), end], axis=1)
        #output = tf.gather_nd(output, gather_idcs)
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
