import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append('../')

from src.experiment import Experiment
from src.tools import process_results


try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

runs = 1
timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'pen_stroke_small'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': timit_dataset,
                        'tr': {'in_seq_len': 15,
                               'out_seq_len': 5,
                               'zero_padding': 2,
                               'mini_batch_mode': True,
                               'batch_size': 1000},
                        'va': {'in_seq_len': 15,
                               'out_seq_len': 5,
                               'zero_padding': 2,
                               'mini_batch_mode': True,
                               'batch_size': 1000}}

input_config = {'layer_type': 'input'}

hidden_1_config = {'layer_type': 'fc',
                   'var_scope': 'fc_1',
                   'init_config': {'w': 'xavier', 'b': 'all_zero'},
                   'act_func': tf.nn.relu,
                   'is_recurrent': True,
                   'is_output': False,
                   'regularization': {'mode': 'l2',
                                      'strength': 0}}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'is_output': False,
                   'regularization': {'mode': 'zoneout',
                                      'state_zo_prob': .5,
                                      'output_zo_prob': 0.35}}

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'init_config': {'w': 'xavier', 'b': 'all_zero'},
                 'is_recurrent': False,
                 'is_output': True,
                 'regularization': {'mode': 'l2',
                                    'strength': 0.02}}

rnn_config = {'layout': [13, 15, 54],
              'layer_configs': [input_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification'}

training_config = {'learning_rate': 0.1,
                   'max_epochs': 1000,
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': np.arange(10, 30),
                            'out_seq_len': np.arange(5, 25),
                            'epochs': np.arange(20) * 5}}

info_config = {'calc_performance_every': 1,
               'include_pred': False,
               'include_out': False}

result_config = {'save_results': False,
                 'filename': '../numerical_results/test',
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config))
print('----------------------------')
print(result_dicts[0])
process_results(result_config, result_dicts)


