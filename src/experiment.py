import tensorflow as tf
from src.data_loader import load_dataset
from src.data_tools import LabelledData
from src.rnn import RNN


class Experiment:
    def __init__(self):
        tf.reset_default_graph()
        self.rnn = None
        self.labelled_data = None
        self.labelled_data_config = None
        self.rnn_config = None

    def create_rnn(self, rnn_config, labelled_data):
        if self.rnn_config is None:
            self.rnn_config = rnn_config
        self.rnn = RNN(rnn_config, labelled_data)

        self.rnn.create_training_graph()
        self.rnn.create_validation_graph()

    def train(self, rnn_config, labelled_data_config, training_config, info_config):
        data_dict = load_dataset(labelled_data_config)
        self.remove_data(data_dict, 445)
        labelled_data = LabelledData(labelled_data_config, data_dict['x_tr'].shape, data_dict['y_tr'].shape,
                                     data_dict['x_va'].shape, data_dict['y_va'].shape)
        self.create_rnn(rnn_config, labelled_data)
        self.labelled_data = labelled_data
        self.labelled_data_config = labelled_data_config

        # Initialize dictionary where all results are stored
        result_dict = {'tr': {'outs': [], 'loss': [], 'preds': [], 'accs': [], 'epochs': []},
                       'va': {'outs': [], 'loss': [], 'preds': [], 'accs': [], 'epochs': []}}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(labelled_data.load_tr_set_op, feed_dict={labelled_data.x_tr_placeholder: data_dict['x_tr'],
                                                              labelled_data.y_tr_placeholder: data_dict['y_tr']})
            sess.run(labelled_data.load_va_set_op, feed_dict={labelled_data.x_va_placeholder: data_dict['x_va'],
                                                              labelled_data.y_va_placeholder: data_dict['y_va']})

            for epoch in range(training_config['max_epochs']):
                if epoch % info_config['calc_performance_every'] == 0:
                    tr_acc, tr_loss, va_acc, va_loss = self.store_performance(sess, info_config, result_dict, epoch)
                    print('{} | TrAcc: {:6.4f}, TrLoss: {:8.5f}, VaAcc: {:6.4f}, VaLoss: {:8.5f}'
                          .format(epoch, tr_acc, tr_loss, va_acc, va_loss))

                if labelled_data_config['mini_batch_mode']:
                    sess.run(labelled_data.shuffle_tr_samples)
                    for minibatch_idx in range(labelled_data.n_tr_minibatches):
                        sess.run(self.rnn.train_op, feed_dict={self.rnn.learning_rate: training_config['learning_rate'],
                                                               labelled_data.batch_counter: minibatch_idx})
                else:
                    sess.run(self.rnn.train_op, feed_dict={self.rnn.learning_rate: training_config['learning_rate']})
                # print(sess.run(self.rnn.gradients, feed_dict={self.rnn.labelled_data.is_validation: False}))

        return result_dict

    def retrieve_performance(self, sess):
        if self.labelled_data_config['mini_batch_mode']:
            tr_cum_loss = 0
            tr_cum_acc = 0
            va_cum_loss = 0
            va_cum_acc = 0

            for minibatch_idx in range(self.labelled_data.n_tr_minibatches):
                loss, acc = sess.run([self.rnn.tr_loss, self.rnn.tr_acc],
                                     feed_dict={self.labelled_data.batch_counter: minibatch_idx})
                tr_cum_loss += loss
                tr_cum_acc += acc
            for minibatch_idx in range(self.labelled_data.n_va_minibatches):
                loss, acc = sess.run([self.rnn.va_loss, self.rnn.va_acc],
                                     feed_dict={self.labelled_data.batch_counter: minibatch_idx})
                va_cum_loss += loss
                va_cum_acc += acc
            tr_acc = tr_cum_acc / self.labelled_data.n_tr_minibatches
            tr_loss = tr_cum_loss / self.labelled_data.n_tr_minibatches
            va_acc = va_cum_acc / self.labelled_data.n_va_minibatches
            va_loss = va_cum_loss / self.labelled_data.n_va_minibatches
        else:
            loss, acc = sess.run([self.rnn.tr_loss, self.rnn.tr_acc])
            tr_loss = loss
            tr_acc = acc
            loss, acc = sess.run([self.rnn.va_loss, self.rnn.va_acc])
            va_loss = loss
            va_acc = acc

        return tr_loss, tr_acc, va_loss, va_acc

    def store_performance(self, sess, info_config, result_dict, epoch):
        tr_loss, tr_acc, va_loss, va_acc = self.retrieve_performance(sess)
        self.update_result_dict(result_dict, info_config, 'tr', None, tr_acc, None, tr_loss, epoch)
        self.update_result_dict(result_dict, info_config, 'va', None, va_acc, None, va_loss, epoch)
        return tr_acc, tr_loss, va_acc, va_loss

    def update_result_dict(self, result_dict, info_config, dict_key, out, acc, pred, loss, epoch):
        if info_config['include_out']:
            result_dict[dict_key]['outs'].append(out)
        if self.rnn_config['output_type'] == 'classification':
            result_dict[dict_key]['accs'].append(acc)
            if info_config['include_pred']:
                result_dict[dict_key]['preds'].append(pred)

        result_dict[dict_key]['loss'].append(loss)
        result_dict[dict_key]['epochs'].append(epoch)

    def remove_data(self, data_dict, n_samples):
        data_dict['x_tr'] = data_dict['x_tr'][:n_samples, :, :]
        data_dict['y_tr'] = data_dict['y_tr'][:n_samples, :, :]
        data_dict['x_va'] = data_dict['x_va'][:n_samples, :, :]
        data_dict['y_va'] = data_dict['y_va'][:n_samples, :, :]
        data_dict['tr_seqlen'] = data_dict['tr_seqlen'][:n_samples]
        data_dict['va_seqlen'] = data_dict['va_seqlen'][:n_samples]


