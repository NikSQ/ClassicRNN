import tensorflow as tf
from src.data_loader import load_dataset
from src.data_tools import LabelledData
from src.rnn import RNN


class Experiment:
    def __init__(self, rnn_config):
        self.rnn = RNN(rnn_config)

    def train(self, labelled_data_config, training_config):
        data_dict = load_dataset(labelled_data_config)
        labelled_data = LabelledData(data_dict['x_tr'].shape[1:], data_dict['y_tr'].shape[1:],
                                     data_dict['x_tr'].shape[0], data_dict['x_va'].shape[0])
        self.rnn.create_training_graph(labelled_data)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(labelled_data.load_tr_set_op, feed_dict={labelled_data.x_placeholder: data_dict['x_tr'],
                                                              labelled_data.y_placeholder: data_dict['y_tr']})
            sess.run(labelled_data.load_va_set_op, feed_dict={labelled_data.x_placeholder: data_dict['x_va'],
                                                              labelled_data.y_placeholder: data_dict['y_va']})
            sess.run(self.rnn.train_op, feed_dict={self.rnn.labelled_data.is_validation: False,
                                                   self.rnn.learning_rate: training_config['learning_rate']})


