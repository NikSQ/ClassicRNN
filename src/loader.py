from scipy.io import loadmat
import numpy as np
from src.preprocessing import extract_seqs
import itertools


filenames = {'red_penstroke': '../datasets/mnist_pen_strokes/mnist_pen_stroke_5000_1000.mat',
             'timit_tr_small_0': '../datasets/timit/tr_20_0.mat',
             'timit_va_small_0': '../datasets/timit/va_20_0.mat',
             'timit_tr_s_0': '../../datasets/timit/tr_s2l10_50_0.mat',
             'timit_va_s_0': '../../datasets/timit/va_s2l10_20_0.mat',
             'timit_tr_l_0': '../../datasets/timit/tr_s10l25_100_0.mat',
             'timit_va_l_0': '../../datasets/timit/va_s10l25_40_0.mat',
             'penstroke_tr': '../datasets/mnist_pen_strokes/mps_full1_tr.mat',
             'penstroke_va': '../datasets/mnist_pen_strokes/mps_full1_va.mat',
             'penstroke_te': '../datasets/mnist_pen_strokes/mps_full1_te.mat'}


def load_dataset(l_data_config):
    entry = l_data_config['dataset']
    if entry == 'penstroke':
        data_dict = load_penstroke(l_data_config)
    elif entry.startswith('timit'):
        data_dict = load_timit(l_data_config)
    elif type(entry) is str:
        data_dict = load_single_file(l_data_config)
    else:
        data_dict = load_files(l_data_config)

    return data_dict


def load_timit(l_data_config):
    if l_data_config['dataset'] == 'timit_s':
        n_tr_phonems = 10000
        n_va_phonems = 1000
        n_te_phonems = 1500
    timit_path = '../../datasets/timit/'
    n_tv_phonems = n_tr_phonems + n_va_phonems
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
    permuted_indices = np.random.permutation(np.arange(n_tv_phonems))
    tr_speaker_idc = permuted_indices[:n_tr_phonems]
    tr_idc = tr_speaker_idc
    va_idc = permuted_indices[n_tr_phonems:n_tr_phonems + n_va_phonems]
    data_dict['tr']['x'], data_dict['tr']['y'], data_dict['tr']['end'] = \
        extract_seqs(partial_dict['x'][tr_idc], partial_dict['y'][tr_idc],
                     partial_dict['seqlen'][tr_idc], l_data_config['tr'])

    data_dict['va']['x'], data_dict['va']['y'], data_dict['va']['end_time'] = \
        extract_seqs(partial_dict['x'][va_idc], partial_dict['y'][va_idc],
                     partial_dict['seqlen'][va_idc], l_data_config['va'])

    data_dict['te'] = {'end': [], 'x': [], 'y': []}
    n_phonems = 0
    for mat in range(7):
        partial_set = loadmat(timit_path + 'te' + str(mat + 1) + '.mat')
        seqlen = np.squeeze(partial_set['seqlen']).astype(np.int32)
        x, y, end_time = extract_seqs(partial_set['x'], partial_set['y'], seqlen, l_data_config['te'])
        data_dict['te']['x'].append(x)
        data_dict['te']['y'].append(y)
        data_dict['te']['end'].append(end_time)
        n_phonems += partial_set['x'].shape[0]
        if n_phonems > n_te_phonems:
            break

    data_dict['te']['x'] = np.concatenate(data_dict['te']['x'], axis=0)
    data_dict['te']['y'] = np.concatenate(data_dict['te']['y'], axis=0)
    data_dict['te']['end'] = np.concatenate(data_dict['te']['end'], axis=0)
    te_idc = np.random.permutation(np.arange(n_te_phonems))
    data_dict['te']['x'] = data_dict['te']['x'][te_idc]
    data_dict['te']['y'] = data_dict['te']['y'][te_idc]
    data_dict['te']['end'] = data_dict['te']['end'][te_idc]

    return data_dict


def load_single_file(l_data_config):
    dataset = loadmat(filenames[l_data_config['dataset']])
    data_dict = {'tr': {}, 'va': {}}
    data_dict['tr']['seqlen'] = np.squeeze(dataset['tr_seqlen']).astype(np.int32)
    data_dict['va']['seqlen'] = np.squeeze(dataset['va_seqlen']).astype(np.int32)
    data_dict['tr']['x'], data_dict['tr']['y'] = extract_seqs(dataset['x_tr'], dataset['y_tr'],
                                                              data_dict['tr']['seqlen'], l_data_config['tr'])
    data_dict['va']['x'], data_dict['va']['y'] = extract_seqs(dataset['x_va'], dataset['y_va'],
                                                              data_dict['va']['seqlen'], l_data_config['va'])
    print(data_dict['tr']['x'].shape)
    print(data_dict['va']['x'].shape)
    return data_dict


def load_files(l_data_config):
    tr_dataset = loadmat(filenames[l_data_config['dataset'][0]])
    data_dict = {'tr': {}, 'va': {}}
    data_dict['tr']['seqlen'] = np.squeeze(tr_dataset['seqlen']).astype(np.int32)
    data_dict['tr']['x'], data_dict['tr']['y'] = extract_seqs(tr_dataset['x'], tr_dataset['y'],
                                                              tr_dataset['tr_seqlen'], l_data_config['tr'])

    va_dataset = loadmat(filenames[l_data_config['dataset'][1]])
    data_dict['va']['seqlen'] = np.squeeze(va_dataset['seqlen']).astype(np.int32)
    data_dict['va']['x'], data_dict['va']['y'] = extract_seqs(va_dataset['x'], va_dataset['y'],
                                                              va_dataset['va_seqlen'], l_data_config['va'])
    return data_dict


def load_penstroke(l_data_config):
    keys = ['tr', 'va', 'te']
    data_dict = dict()
    for data_key in keys:
        if data_key in l_data_config.keys():
            dataset = loadmat(filenames['penstroke_' + data_key])
            data_dict[data_key] = dict()
            data_dict[data_key]['x'], data_dict[data_key]['y'], data_dict[data_key]['end'] = \
                extract_seqs(dataset['x'], dataset['y'], np.squeeze(dataset['seqlen']).astype(np.int32),
                             l_data_config[data_key])
            print(data_key)
            print(data_dict[data_key]['x'].shape)
    return data_dict

