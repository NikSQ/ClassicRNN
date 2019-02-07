import sys
import os
import tensorflow as tf
import numpy as np
import copy

sys.path.append('../')

from src.experiment import Experiment
from src.tools import process_results


try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except KeyError:
    print('NO SLURM TASK ID FOUND')
    print('Using ID 0 instead')
    task_id = 0

filename = 'learning_rate'
runs = 1
timit_dataset = ['timit_tr_small_0', 'timit_va_small_0']
timit_s_dataset = ['timit_tr_s_0', 'timit_va_s_0']
timit_l_dataset = ['timit_tr_l_0', 'timit_va_l_0']
penstroke_dataset = 'penstroke'
# timit: 13 -> 54
# penstroke: 4 -> 10
labelled_data_config = {'dataset': penstroke_dataset,
                        'tr': {'in_seq_len': 40,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'mini_batch_mode': False,
                               'batch_size': 5000,
                               'extract_seqs': False},
                        'va': {'in_seq_len': 40,
                               'out_seq_len': 1,
                               'zero_padding': 0,
                               'mini_batch_mode': False,
                               'batch_size': 5000, 
                               'extract_seqs': False}}

input_config = {'layer_type': 'input'}

p_state = [.1, .1, .1, .3, .3, .3, .7, .7, .7, .9, .9, .9] # .3
p_out = [.05, .1, .5, .05, .1, .5, .05, .1, .5, .05, .1, .5] # .1
p_state_c = [.1, .1, .3, .3, .1, .1, .3, .3, .1, .1, .3, .3] # .3
p_out_c = [.05, .1, .05, .1, .05, .1, .05, .1, .05, .1, .05, .1 ] # .05
f_init = [{'w': 'all_zero', 'b': 'all_one'}, {'w': 'xavier', 'b': 'all_one'}, {'w': 'all_zero', 'b': 'all_zero'}, {'w': 'xavier', 'b': 'all_zero'}]

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': f_init[0],
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'is_output': False,
                   'regularization': {'mode': 'zoneout',
                                      'state_zo_prob': p_state_c[2],
                                      'output_zo_prob': p_out_c[2]}}
hidden_1_config = copy.deepcopy(hidden_2_config)
hidden_1_config['var_scope'] = 'lstm_0'

l2_strenghts = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.] #0.01
l2_strenghts_c = [0.003, 0.003, 0.003, 0.003, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03] # 0.003
output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'init_config': {'w': 'xavier', 'b': 'all_zero'},
                 'is_recurrent': False,
                 'is_output': True,
                 'regularization': {'mode': 'l2',
                                    'strength': l2_strenghts_c[2]}}

#n_lstm = [5, 10, 15, 20, 30, 40, 50, 65, 85, 110, 150, 180] # 150, 180
n_lstm1 = [20, 20, 20, 40, 40, 40, 60, 60, 60] # 60
n_lstm2 = [20, 40, 60, 20, 40, 60, 20, 40, 60] # 60
rnn_config = {'layout': [4, 60, 60, 10],
              'layer_configs': [input_config, hidden_1_config, hidden_2_config, output_config],
              'gradient_clip_value': .5,
              'output_type': 'classification'}

# learning rates .1  is already too high. look for <= .01
learning_rate = [.1, .01]
training_config = {'learning_rate': learning_rate[task_id],
                   'mode': {'name': 'inc_lengths',
                            'in_seq_len': [1, 2, 4, 8, 30],
                            'out_seq_len': [1, 1, 2, 4, 5],
                            'zero_padding': [0, 0, 0, 2, 23],
                            'min_errors': [0, 0, 0, 0, 0.],
                            'max_epochs': [2, 10, 30, 50, 2000]},
                   'task_id': task_id}
training_config['mode'] = {'name': 'classic', 'min_error': 0, 'max_epochs': 600}

info_config = {'calc_performance_every': 1,
               'include_pred': False,
               'include_out': False}

result_config = {'save_results': False,
                 'filename': '../numerical_results/' + filename + '_' + str(task_id),
                 'plot_results': False,
                 'print_final_stats': True}

result_dicts = []
print(training_config)
print(labelled_data_config)
print(rnn_config)
print(info_config)
for run in range(runs):
    experiment = Experiment()
    result_dicts.append(experiment.train(rnn_config, labelled_data_config, training_config, info_config))
print('----------------------------')
#print(result_dicts[0])
process_results(result_config, result_dicts)

