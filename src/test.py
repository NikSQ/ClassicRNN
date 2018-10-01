import sys
import tensorflow as tf

sys.path.append('../')

from src.experiment import Experiment

runs=5

labelled_data_config = {'dataset': 'pen_stroke_small',
                        'in_seq_len': 20,
                        'out_seq_len': 6,
                        'do_zero_padding': False}

input_config = {'layer_type': 'input'}

hidden_1_config = {'layer_type': 'fc',
                   'var_scope': 'fc_1',
                   'init_config': {'w': 'xavier', 'b': 'all_zero'},
                   'act_func': tf.nn.relu,
                   'is_recurrent': True,
                   'is_output': False}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'regularization': {'mode': 'no_reg',
                                      'state_zo_prob': 0.5,
                                      'output_zo_prob': 0.05}}

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'init_config': {'w': 'xavier', 'b': 'all_zero'},
                 'is_recurrent': False,
                 'is_output': True}

rnn_config = {'layout': [4, 20, 20, 10],
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': 1.0,
              'output_type': 'classification'}

training_config = {'learning_rate': 0.1,
                   'max_epochs': 10}

info_config = {'calc_tr_performance_every': 1,
               'calc_va_performance_every': 1,
               'include_pred': False,
               'include_out': False}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config))


