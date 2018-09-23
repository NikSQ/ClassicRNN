import tensorflow as tf


class DataContainer:
    def __init__(self, x_shape, y_shape, n_tr_samples, n_va_samples):
        self.n_tr_samples = n_tr_samples
        self.n_va_samples = n_va_samples
        self.x_shape = x_shape
        self.y_shape = y_shape

        with tf.variable_scope('data'):
            self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=x_shape)
            self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=y_shape)
            self.x_tr = tf.get_variable(name='x_tr', shape=(n_tr_samples,) + x_shape, dtype=tf.float32)
            self.y_tr = tf.get_variable(name='y_tr', shape=(n_tr_samples,) + y_shape, dtype=tf.float32)
            self.x_va = tf.get_variable(name='x_va', shape=(n_va_samples,) + x_shape, dtype=tf.float32)
            self.y_va = tf.get_variable(name='y_va', shape=(n_va_samples,) + y_shape, dtype=tf.float32)

            assign_x_tr_op = tf.assign(self.x_tr, self.x_placeholder)
            assign_y_tr_op = tf.assign(self.y_tr, self.y_placeholder)
            self.load_tr_set_op = tf.group(*[assign_x_tr_op, assign_y_tr_op])

            assign_x_va_op = tf.assign(self.x_va, self.x_placeholder)
            assign_y_va_op = tf.assign(self.y_va, self.y_placeholder)
            self.load_va_set_op = tf.group(*[assign_x_va_op, assign_y_va_op])

