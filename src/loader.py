from scipy.io import loadmat
import numpy as np
from src.preprocessing import extract_seqs


filenames = {'red_penstroke': '../datasets/mnist_pen_strokes/mnist_pen_stroke_5000_1000.mat',
             'timit_tr_small_0': '../datasets/timit/tr_20_0.mat',
             'timit_va_small_0': '../datasets/timit/va_20_0.mat',
             'timit_tr_s_0': '../datasets/timit/tr_s2l10_50_0.mat',
             'timit_va_s_0': '../datasets/timit/va_s2l10_20_0.mat',
             'timit_tr_l_0': '../datasets/timit/tr_s10l25_100_0.mat',
             'timit_va_l_0': '../datasets/timit/va_s10l25_40_0.mat',
             'penstroke_tr': '../datasets/mnist_pen_strokes/mps_full1_tr.mat',
             'penstroke_va': '../datasets/mnist_pen_strokes/mps_full1_va.mat',
             'penstroke_te': '../datasets/mnist_pen_strokes/mps_full1_te.mat'}


def load_dataset(l_data_config):
    entry = l_data_config['dataset']
    if entry == 'penstroke':
        data_dict = load_penstroke(l_data_config)
    elif type(entry) is str:
        data_dict = load_single_file(l_data_config)
    else:
        data_dict = load_files(l_data_config)

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
    data_dict = {'tr': {}, 'va': {}, 'te': {}}
    for data_key in data_dict.keys():
        dataset = loadmat(filenames['penstroke_' + data_key])
        data_dict[data_key]['seqlen'] = np.squeeze(dataset['seqlen']).astype(np.int32)
        data_dict[data_key]['x'], data_dict[data_key]['y'] = extract_seqs(dataset['x'], dataset['y'],
                                                                          data_dict[data_key]['seqlen'],
                                                                          l_data_config['tr'])
    return data_dict

