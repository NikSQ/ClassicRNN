import tensorflow as tf
from src.loader import load_dataset
from src.data_tools import LabeledData
from src.rnn import RNN
from src.tools import print_config, set_momentum
from tensorflow.python import debug as tf_debug


class Experiment:
    def __init__(self, rnn_config, l_data_config, training_config, info_config):
        self.rnn = None
        self.l_data = None
        self.l_data_config = l_data_config
        self.training_config = training_config
        self.rnn_config = rnn_config
        self.info_config = info_config
        self.data_dict = None

    def create_rnn(self, rnn_config, l_data, l_data_config):
        if self.rnn_config is None:
            self.rnn_config = rnn_config
        set_momentum(self.training_config['batchnorm']['momentum'])
        self.rnn = RNN(rnn_config, self.training_config, l_data)

        self.rnn.create_training_graph()
        self.rnn.create_evaluation_graph('va')
        self.rnn.create_evaluation_graph('te')
        self.l_data = l_data
        self.l_data_config = l_data_config

    def create_modificated_model(self, data_mod_config):
        incremental_idx = data_mod_config['session_idx']
        self.l_data_config['tr']['in_seq_len'] = data_mod_config['in_seq_len'][incremental_idx]
        self.l_data_config['tr']['max_truncation'] = data_mod_config['max_truncation'][incremental_idx]
        self.data_dict = load_dataset(self.l_data_config)
        l_data = LabeledData(self.l_data_config, self.data_dict)
        self.create_rnn(self.rnn_config, l_data, self.l_data_config)

    def train(self):
        print_config(self.training_config, self.l_data_config, self.rnn_config)
        temp_model_path = '../models/temp' + str(self.training_config['task_id'])

        if self.training_config['mode']['name'] == 'inc_lengths':
            n_sessions = len(self.training_config['mode']['in_seq_len'])
        elif self.training_config['mode']['name'] == 'classic':
            n_sessions = 1
        else:
            raise Exception('training mode not understood')

        # Initialize dictionary where all results are stored
        result_dict = {'tr': {'loss': [], 'accs': []},
                       'va': {'loss': [], 'accs': []},
                       'te': {'loss': [], 'accs': []},
                       'epochs': []}

        current_epoch = 0
        learning_rate = self.training_config['learning_rate']
        for session_idx in range(n_sessions):
            tf.reset_default_graph()
            if self.training_config['mode']['name'] == 'inc_lengths':
                self.training_config['mode']['session_idx'] = session_idx
                max_epochs = self.training_config['mode']['max_epochs'][session_idx]
                min_error = self.training_config['mode']['min_errors'][session_idx]
                self.create_modificated_model(self.training_config['mode'])
            elif self.training_config['mode']['name'] == 'classic':
                self.data_dict = load_dataset(self.l_data_config)

                l_data = LabeledData(self.l_data_config, self.data_dict)
                self.create_rnn(self.rnn_config, l_data, self.l_data_config)
                max_epochs = self.training_config['mode']['max_epochs']
                min_error = self.training_config['mode']['min_error']

            model_saver = tf.train.Saver(tf.trainable_variables())
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if session_idx != 0:
                    model_saver.restore(sess, temp_model_path)
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
                for epoch in range(max_epochs):
                    if epoch % self.info_config['calc_performance_every'] == 0:
                        self.compute_performance(sess, result_dict, epoch)
                        print('{:3}, {:2} | TrAcc: {:6.4f}, TrLoss: {:8.5f}, VaAcc: {:6.4f}, VaLoss: {:8.5f}'
                              .format(current_epoch, session_idx, result_dict['tr']['accs'][-1],
                                      result_dict['tr']['loss'][-1], result_dict['va']['accs'][-1],
                                      result_dict['va']['loss'][-1]))
                        if result_dict['tr']['loss'][-1] < min_error:
                            break

                    if (current_epoch + 1) % self.training_config['learning_rate_tau'] == 0:
                        learning_rate /= 2

                    if self.l_data_config['tr']['minibatch_mode']:
                        sess.run(self.l_data.data['tr']['shuffle'])
                    for minibatch_idx in range(self.l_data.data['tr']['n_minibatches']):
                        sess.run(self.rnn.train_op, feed_dict={self.rnn.learning_rate: learning_rate,
                                                               self.l_data.batch_counter: minibatch_idx,
                                                               self.rnn.is_training: True})
                    current_epoch += 1
                    # print(sess.run(self.rnn.gradients, feed_dict={self.rnn.labelled_data.is_validation: False}))
                model_saver.save(sess, temp_model_path)
        return result_dict

    def compute_performance(self, sess, result_dict, epoch):
        for data_key in self.l_data.data.keys():
            cum_acc = 0
            cum_loss = 0
            for minibatch_idx in range(self.l_data.data[data_key]['n_minibatches']):
                acc, loss = sess.run([self.rnn.metrics[data_key]['acc'], self.rnn.metrics[data_key]['loss']],
                                     feed_dict={self.l_data.batch_counter: minibatch_idx, self.rnn.is_training: False})
                cum_acc += acc
                cum_loss += loss
            acc = cum_acc / self.l_data.data[data_key]['n_minibatches']
            loss = cum_loss / self.l_data.data[data_key]['n_minibatches']
            result_dict[data_key]['accs'].append(acc)
            result_dict[data_key]['loss'].append(loss)
        result_dict['epochs'].append(epoch)

