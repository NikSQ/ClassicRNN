import sys
import os
import tensorflow as tf

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
labelled_data_config = {'dataset': 'pen_stroke_small',
                        'in_seq_len': 50,
                        'out_seq_len': 1,
                        'do_zero_padding': False,
                        'batch_mode': True,
                        'batch_size': 10}

input_config = {'layer_type': 'input'}

hidden_1_config = {'layer_type': 'fc',
                   'var_scope': 'fc_1',
                   'init_config': {'w': 'xavier', 'b': 'all_zero'},
                   'act_func': tf.nn.relu,
                   'is_recurrent': True,
                   'is_output': False,
                   'regularization': {'mode': None}}

hidden_2_config = {'layer_type': 'lstm',
                   'var_scope': 'lstm_1',
                   'init_config': {'f': {'w': 'all_zero', 'b': 'all_one'},
                                   'i': {'w': 'xavier', 'b': 'all_zero'},
                                   'c': {'w': 'xavier', 'b': 'all_zero'},
                                   'o': {'w': 'xavier', 'b': 'all_zero'}},
                   'is_recurrent': True,
                   'is_output': False,
                   'regularization': {'mode': 'zoneout',
                                      'state_zo_prob': 0.5,
                                      'output_zo_prob': 0.05}}

output_config = {'layer_type': 'fc',
                 'var_scope': 'output_layer',
                 'init_config': {'w': 'xavier', 'b': 'all_zero'},
                 'is_recurrent': False,
                 'is_output': True,
                 'regularization': {'mode': None}}

rnn_config = {'layout': [4, 20, 10],
              'layer_configs': [input_config, hidden_2_config, output_config],
              'gradient_clip_value': .4,
              'output_type': 'classification'}

training_config = {'learning_rate': 0.1,
                   'max_epochs': 300}

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


