import tensorflow as tf
from src.data_loader import load_dataset
from src.data_tools import LabelledData
from src.rnn import RNN


class Experiment:
    def __init__(self):
        tf.reset_default_graph()
        self.rnn = None
        self.rnn_config = None

    def create_rnn(self, rnn_config):
        if self.rnn_config is None:
            self.rnn_config = rnn_config
        self.rnn = RNN(rnn_config)

    def train(self, rnn_config, labelled_data_config, training_config, info_config):
        data_dict = load_dataset(labelled_data_config)
        labelled_data = LabelledData(data_dict['x_tr'].shape[1:], data_dict['y_tr'].shape[1:],
                                     data_dict['x_tr'].shape[0], data_dict['x_va'].shape[0])
        self.create_rnn(rnn_config)
        self.rnn.create_training_graph(labelled_data)

        # Initialize dictionary where all results are stored
        result_dict = {'tr': {'outs': [], 'loss': [], 'preds': [], 'accs': [], 'epochs': []},
                       'va': {'outs': [], 'loss': [], 'preds': [], 'accs': [], 'epochs': []}}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(labelled_data.load_tr_set_op, feed_dict={labelled_data.x_placeholder: data_dict['x_tr'],
                                                              labelled_data.y_placeholder: data_dict['y_tr']})
            sess.run(labelled_data.load_va_set_op, feed_dict={labelled_data.x_placeholder: data_dict['x_va'],
                                                              labelled_data.y_placeholder: data_dict['y_va']})

            for epoch in range(training_config['max_epochs']):
                if epoch % info_config['calc_tr_performance_every'] == 0:
                    result_dict = self.retrieve_performance(sess, False, info_config, result_dict, epoch)
                if epoch % info_config['calc_va_performance_every'] == 0:
                    result_dict = self.retrieve_performance(sess, True, info_config, result_dict, epoch)

                sess.run(self.rnn.train_op, feed_dict={self.rnn.labelled_data.is_validation: False,
                                                       self.rnn.learning_rate: training_config['learning_rate']})
                print(sess.run(self.rnn.gradients, feed_dict={self.rnn.labelled_data.is_validation: False}))

        return result_dict

    def retrieve_performance(self, sess, is_validation, info_config, result_dict, epoch):
        if self.rnn_config['output_type'] == 'classification':
            out, pred, loss, acc = sess.run([self.rnn.output, self.rnn.prediction, self.rnn.loss,
                                             self.rnn.accuracy],
                                            feed_dict={self.rnn.labelled_data.is_validation: is_validation})
        else:
            out, loss = sess.run([self.rnn.output, self.rnn.loss],
                                 feed_dict={self.rnn.labelled_data.is_validation: is_validation})

        if is_validation:
            dict_key = 'va'
        else:
            dict_key = 'tr'

        if info_config['include_out']:
            result_dict[dict_key]['outs'].append(out)
        if self.rnn_config['output_type'] == 'classification':
            result_dict[dict_key]['accs'].append(acc)
            if info_config['include_pred']:
                result_dict[dict_key]['preds'].append(pred)

        result_dict[dict_key]['loss'].append(loss)
        result_dict[dict_key]['epochs'].append(epoch)
        print('{} | Validation: {}, Accuracy: {}, Loss: {}'.format(epoch, is_validation, acc, loss))
        return result_dict


