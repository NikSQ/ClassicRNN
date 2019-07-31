import tensorflow as tf
import numpy as np
from src.logging_tools import get_logger




class LabelledData:
    def __init__(self, labelled_data_config, x_tr_shape, y_tr_shape, x_va_shape, y_va_shape, x_te_shape, y_te_shape):
        self.x_tr_shape = x_tr_shape
        self.y_tr_shape = y_tr_shape
        self.x_va_shape = x_va_shape
        self.y_va_shape = y_va_shape
        self.x_te_shape = x_te_shape
        self.y_te_shape = y_te_shape

        with tf.variable_scope('labelled_data'):
            self.x_tr_placeholder = tf.placeholder(dtype=tf.float32, shape=x_tr_shape)
            self.y_tr_placeholder = tf.placeholder(dtype=tf.float32, shape=y_tr_shape)
            self.x_va_placeholder = tf.placeholder(dtype=tf.float32, shape=x_va_shape)
            self.y_va_placeholder = tf.placeholder(dtype=tf.float32, shape=y_va_shape)
            self.x_te_placeholder = tf.placeholder(dtype=tf.float32, shape=x_te_shape)
            self.y_te_placeholder = tf.placeholder(dtype=tf.float32, shape=y_te_shape)
            self.x_tr = tf.get_variable(name='x_tr', shape=x_tr_shape, dtype=tf.float32, trainable=False)
            self.y_tr = tf.get_variable(name='y_tr', shape=y_tr_shape, dtype=tf.float32, trainable=False)
            self.x_va = tf.get_variable(name='x_va', shape=x_va_shape, dtype=tf.float32, trainable=False)
            self.y_va = tf.get_variable(name='y_va', shape=y_va_shape, dtype=tf.float32, trainable=False)
            self.x_te = tf.get_variable(name='x_te', shape=x_te_shape, dtype=tf.float32, trainable=False)
            self.y_te = tf.get_variable(name='y_te', shape=y_te_shape, dtype=tf.float32, trainable=False)

            self.end_time_tr_ph = tf.placeholder(dtype=tf.int32, shape=(x_tr_shape[0],))
            self.end_time_va_ph = tf.placeholder(dtype=tf.int32, shape=(x_va_shape[0],))
            self.end_time_te_ph = tf.placeholder(dtype=tf.int32, shape=(x_te_shape[0],))

            self.end_time_tr = tf.get_variable(name='end_tr', dtype=tf.int32, shape=(x_tr_shape[0],))
            self.end_time_va = tf.get_variable(name='end_va', dtype=tf.int32, shape=(x_va_shape[0],))
            self.end_time_te = tf.get_variable(name='end_te', dtype=tf.int32, shape=(x_te_shape[0],))

            assign_x_tr_op = tf.assign(self.x_tr, self.x_tr_placeholder)
            assign_y_tr_op = tf.assign(self.y_tr, self.y_tr_placeholder)
            assign_end_tr_op = tf.assign(self.end_time_tr, self.end_time_tr_ph)
            self.load_tr_set_op = tf.group(*[assign_x_tr_op, assign_y_tr_op, assign_end_tr_op])

            assign_x_va_op = tf.assign(self.x_va, self.x_va_placeholder)
            assign_y_va_op = tf.assign(self.y_va, self.y_va_placeholder)
            assign_end_va_op = tf.assign(self.end_time_va, self.end_time_va_ph)
            self.load_va_set_op = tf.group(*[assign_x_va_op, assign_y_va_op, assign_end_va_op])

            assign_x_te_op = tf.assign(self.x_te, self.x_te_placeholder)
            assign_y_te_op = tf.assign(self.y_te, self.y_te_placeholder)
            assign_end_te_op = tf.assign(self.end_time_te, self.end_time_te_ph)
            self.load_te_set_op = tf.group(*[assign_x_te_op, assign_y_te_op, assign_end_te_op])

            self.batch_counter = tf.placeholder(dtype=tf.int32)

            if labelled_data_config['tr']['mini_batch_mode']:
                self.shuffle_tr_samples, self.x_tr_batch, self.y_tr_batch, self.end_tr_batch, self.n_tr_minibatches = \
                    self.create_minibatch_support(labelled_data_config['tr'], 'tr', self.x_tr_shape, self.x_tr,
                                                  self.y_tr, self.end_time_tr)

                self.x_tr_shape = (labelled_data_config['tr']['batch_size'],) + self.x_tr_shape[1:]
                self.y_tr_shape = (labelled_data_config['tr']['batch_size'],) + self.y_tr_shape[1:]
            else:
                self.x_tr_batch = self.x_tr
                self.y_tr_batch = self.y_tr
                self.end_tr_batch = self.end_time_tr

            if labelled_data_config['va']['mini_batch_mode']:
                self.shuffle_va_samples, self.x_va_batch, self.y_va_batch, self.end_va_batch, self.n_va_minibatches = \
                    self.create_minibatch_support(labelled_data_config['va'], 'va', self.x_va_shape, self.x_va,
                                                  self.y_va, self.end_time_va)
                self.x_va_shape = (labelled_data_config['va']['batch_size'],) + self.x_va_shape[1:]
                self.y_va_shape = (labelled_data_config['va']['batch_size'],) + self.y_va_shape[1:]
            else:
                self.x_va_batch = self.x_va
                self.y_va_batch = self.y_va
                self.end_va_batch = self.end_time_va

            if labelled_data_config['te']['mini_batch_mode']:
                self.shuffle_te_samples, self.x_te_batch, self.y_te_batch, self.end_te_batch, self.n_te_minibatches = \
                    self.create_minibatch_support(labelled_data_config['te'], 'te', self.x_te_shape, self.x_te,
                                                  self.y_te, self.end_time_te)
                self.x_te_shape = (labelled_data_config['te']['batch_size'],) + self.x_te_shape[1:]
                self.y_te_shape = (labelled_data_config['te']['batch_size'],) + self.y_te_shape[1:]
            else:
                self.x_te_batch = self.x_te
                self.y_te_batch = self.y_te
                self.end_te_batch = self.end_time_te

    def create_minibatch_support(self, data_config, dict_key, x_shape, x, y, end_time):
        batch_size = data_config['batch_size']

        # Number of samples is expanded, such that the number of samples is a multiple of the batch size
        n_minibatches = int(np.ceil(float(x_shape[0]) / float(batch_size)))
        n_samples = batch_size * n_minibatches

        # A shuffled list of sample indices. Iterating over the complete list will be one epoch
        sample_list = tf.get_variable(name=dict_key + '_sample_list', shape=n_samples, dtype=tf.int32, trainable=False)
        samples = tf.tile(tf.random_shuffle(tf.range(x_shape[0])), multiples=[int(np.ceil(n_samples / x_shape[0]))])
        shuffle_samples_op = tf.assign(sample_list, samples[:n_samples])

        x_batch = tf.gather(x, indices=samples[self.batch_counter:self.batch_counter+batch_size])
        y_batch = tf.gather(y, indices=samples[self.batch_counter:self.batch_counter+batch_size])
        end_batch = tf.gather(end_time, indices=samples[self.batch_counter:self.batch_counter+batch_size])
        return shuffle_samples_op, x_batch, y_batch, end_batch, n_minibatches

