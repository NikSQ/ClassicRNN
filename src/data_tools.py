import tensorflow as tf
import numpy as np
from src.logging_tools import get_logger


class LabeledData:
    def __init__(self, l_data_config, data_dict):
        self.data = dict()
        self.batch_counter = tf.placeholder(dtype=tf.int32)
        for data_key in data_dict.keys():
            self.data[data_key] = dict()
            x_init = tf.constant_initializer(data_dict[data_key]['x'])
            y_init = tf.constant_initializer(data_dict[data_key]['y'])
            end_init = tf.constant_initializer(data_dict[data_key]['end'], dtype=tf.int32)
            self.data[data_key]['x'] = tf.get_variable(name='x_' + data_key, shape=data_dict[data_key]['x'].shape,
                                                       dtype=tf.float32, trainable=False, initializer=x_init)
            self.data[data_key]['y'] = tf.get_variable(name='y_' + data_key, shape=data_dict[data_key]['y'].shape,
                                                       dtype=tf.float32, trainable=False, initializer=y_init)
            self.data[data_key]['end'] = tf.get_variable(name='end_' + data_key, shape=(data_dict[data_key]['x'].shape[0],),
                                                         dtype=tf.int32, trainable=False, initializer=end_init)

            if l_data_config[data_key]['minibatch_mode']:
                self.data[data_key]['n_minibatches'] = int(np.ceil(float(data_dict[data_key]['x'].shape[0]) /
                                                                   float(l_data_config[data_key]['minibatch_size'])))
                n_samples = self.data[data_key]['n_minibatches'] * l_data_config[data_key]['minibatch_size']

                # A shuffled list of sample indices. Iterating over the complete list will be one epoch
                sample_list = tf.get_variable(name=data_key + '_sample_list', shape=n_samples, dtype=tf.int32, trainable=False)
                samples = tf.tile(tf.random_shuffle(tf.range(data_dict[data_key]['x'].shape[0])), multiples=[int(np.ceil(n_samples / data_dict[data_key]['x'].shape[0]))])
                self.data[data_key]['shuffle'] = tf.assign(sample_list, samples[:n_samples])

                self.data[data_key]['x_batch'] = tf.gather(self.data[data_key]['x'], indices=samples[self.batch_counter:self.batch_counter + l_data_config[data_key]['minibatch_size']])
                self.data[data_key]['y_batch'] = tf.gather(self.data[data_key]['y'], indices=samples[self.batch_counter:self.batch_counter + l_data_config[data_key]['minibatch_size']])
                self.data[data_key]['end_batch'] = tf.gather(self.data[data_key]['end'], indices=samples[self.batch_counter:self.batch_counter + l_data_config[data_key]['minibatch_size']])
                self.data[data_key]['x_shape'] = (l_data_config[data_key]['minibatch_size'],) + \
                                                   data_dict[data_key]['x'].shape[1:]
                self.data[data_key]['y_shape'] = (l_data_config[data_key]['minibatch_size'],) + \
                                                   data_dict[data_key]['y'].shape[1:]
            else:
                self.data[data_key]['n_minibatches'] = 1
                self.data[data_key]['x_batch'] = self.data[data_key]['x']
                self.data[data_key]['y_batch'] = self.data[data_key]['y']
                self.data[data_key]['end_batch'] = self.data[data_key]['end']
                self.data[data_key]['x_shape'] = data_dict[data_key]['x'].shape
                self.data[data_key]['y_shape'] = data_dict[data_key]['y'].shape
