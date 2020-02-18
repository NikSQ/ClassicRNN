from scipy.io import loadmat
import numpy as np
from src.preprocessing import extract_seqs
import itertools


filenames = {'penstroke_tr': '../datasets/mnist_pen_strokes/mps_full1_tr.mat',
             'penstroke_va': '../datasets/mnist_pen_strokes/mps_full1_va.mat',
             'penstroke_te': '../datasets/mnist_pen_strokes/mps_full1_te.mat',
             'sign_language_tr': '../../datasets/sign_language/sign_language_tr.mat',
             'sign_language_va': '../../datasets/sign_language/sign_language_va.mat',
             'sign_language_te': '../../datasets/sign_language/sign_language_te.mat',
             'syn_dataset_tr': '../../datasets/synthetic/syn_tr.mat',
             'syn_dataset_va': '../../datasets/synthetic/syn_va.mat',
             'syn_dataset_te': '../../datasets/synthetic/syn_te.mat'}


def load_dataset(l_data_config):
    entry = l_data_config['dataset']
    if entry.startswith('timit'):
        data_dict = load_timit(l_data_config)
    elif entry.startswith('penstroke'):
        data_dict = get_datadict(l_data_config, 'penstroke')
    elif entry.startswith('sign'):
        data_dict = get_datadict(l_data_config, 'sign_language')
    elif entry.startswith('syn'):
        data_dict = get_datadict(l_data_config, 'syn_dataset')

    return data_dict


def load_timit(l_data_config):
    if l_data_config['dataset'] == 'timit_s':
        n_te_phonems = 20000
        n_tv_phonems = 70000
    timit_path = '../../datasets/timit/'
    data_dict = {'tr': dict(), 'va': dict(), 'te': dict()}

    partial_dict = {'x': [], 'y': [], 'seqlen': []}
    max_len = 0
    speaker_lens = []
    n_phonems = 0
    for mat_idx in range(16):
        path = timit_path + 'tr' + str(mat_idx + 1) + '.mat'
        partial_set = loadmat(path)
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        partial_dict['seqlen'] = np.concatenate([partial_dict['seqlen'], seqlen], axis=0)
        partial_dict['x'].append(partial_set['x'])
        partial_dict['y'].append(partial_set['y'])
        speaker_lens.append(partial_set['x'].shape[0])
        if max_len < partial_set['x'].shape[2]:
            max_len = partial_set['x'].shape[2]
        n_phonems += partial_set['x'].shape[0]
        if n_phonems > n_tv_phonems:
            break
    xs = []
    ys = []
    for x, y in itertools.zip_longest(partial_dict['x'], partial_dict['y']):
        x_shape = x.shape
        xs.append(np.concatenate([np.zeros((x_shape[0], x_shape[1], max_len - x_shape[2])), x], axis=2))
        y_shape = y.shape
        ys.append(np.concatenate([np.zeros((y_shape[0], y_shape[1], max_len - y_shape[2])), y], axis=2))
    partial_dict['x'] = np.concatenate(xs, axis=0)
    partial_dict['y'] = np.concatenate(ys, axis=0)
    permuted_indices = np.random.permutation(np.arange(n_phonems))
    n_tr_phonems = int(len(permuted_indices) * 0.8)
    tr_speaker_idc = permuted_indices[:n_tr_phonems]
    tr_idc = tr_speaker_idc
    va_idc = permuted_indices[n_tr_phonems:]
    data_dict['tr']['x'], data_dict['tr']['y'], data_dict['tr']['end'] = \
        extract_seqs(partial_dict['x'][tr_idc], partial_dict['y'][tr_idc],
                     partial_dict['seqlen'][tr_idc], l_data_config['tr'], False)

    data_dict['va']['x'], data_dict['va']['y'], data_dict['va']['end'] = \
        extract_seqs(partial_dict['x'][va_idc], partial_dict['y'][va_idc],
                     partial_dict['seqlen'][va_idc], l_data_config['va'], False)

    data_dict['te'] = {'end': [], 'x': [], 'y': []}
    n_phonems = 0
    for mat in range(7):
        partial_set = loadmat(timit_path + 'te' + str(mat + 1) + '.mat')
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        x, y, end = extract_seqs(partial_set['x'], partial_set['y'], seqlen, l_data_config['te'], False)
        data_dict['te']['x'].append(x)
        data_dict['te']['y'].append(y)
        data_dict['te']['end'].append(end)
        n_phonems += x.shape[0]
        if n_phonems > n_te_phonems:
            break

    data_dict['te']['x'] = np.concatenate(data_dict['te']['x'], axis=0)
    data_dict['te']['y'] = np.concatenate(data_dict['te']['y'], axis=0)
    data_dict['te']['end'] = np.concatenate(data_dict['te']['end'], axis=0)
    te_idc = np.random.permutation(np.arange(n_phonems))
    data_dict['te']['x'] = data_dict['te']['x'][te_idc]
    data_dict['te']['y'] = data_dict['te']['y'][te_idc]
    data_dict['te']['end'] = data_dict['te']['end'][te_idc]
    print(data_dict['tr']['x'].shape)
    print(data_dict['va']['x'].shape)
    print(data_dict['te']['x'].shape)
    return data_dict


def get_datadict(l_data_config, name):
    keys = ['tr', 'va', 'te']
    data_dict = dict()
    for data_key in keys:
        if data_key in l_data_config.keys():
            dataset = loadmat(filenames[name + '_' + data_key])
            data_dict[data_key] = dict()
            data_dict[data_key]['x'], data_dict[data_key]['y'], data_dict[data_key]['end'] = \
                extract_seqs(dataset['x'], dataset['y'], np.squeeze(dataset['seqlen']).astype(np.int32),
                             l_data_config[data_key], remove_bias=l_data_config['remove_bias'])
            print(data_key)
            print(data_dict[data_key]['x'].shape)
    return data_dict


