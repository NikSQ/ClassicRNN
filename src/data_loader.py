from scipy.io import loadmat
import numpy as np
from src.data_tools import extract_seqs


filenames = {'pen_stroke_small': '../datasets/mnist_pen_strokes/mnist_pen_stroke_5000_1000.mat',
             'timit_tr_small_0': '../datasets/timit/tr_20_0.mat',
             'timit_va_small_0': '../datasets/timit/va_20_0.mat',
             'timit_tr_s_0': '../datasets/timit/tr_s2l10_50_0.mat',
             'timit_va_s_0': '../datasets/timit/va_s2l10_20_0.mat',
             'timit_tr_l_0': '../datasets/timit/tr_s10l25_100_0.mat',
             'timit_va_l_0': '../datasets/timit/va_s10l25_40_0.mat'}


def load_dataset(labelled_data_config):
    entry = labelled_data_config['dataset']
    if type(entry) is str:
        return load_single_file(labelled_data_config)
    else:
        return load_files(labelled_data_config)


def load_single_file(labelled_data_config):
    data_dict = loadmat(filenames[labelled_data_config['dataset']])
    data_dict['tr_seqlen'] = np.squeeze(data_dict['tr_seqlen']).astype(np.int32)
    data_dict['va_seqlen'] = np.squeeze(data_dict['va_seqlen']).astype(np.int32)
    data_dict['x_tr'], data_dict['y_tr'] = extract_seqs(data_dict['x_tr'], data_dict['y_tr'], data_dict['tr_seqlen'],
                                                        labelled_data_config)
    data_dict['x_va'], data_dict['y_va'] = extract_seqs(data_dict['x_va'], data_dict['y_va'], data_dict['va_seqlen'],
                                                        labelled_data_config)
    return data_dict


def load_files(labelled_data_config):
    data_dict_tr = loadmat(filenames[labelled_data_config['dataset'][0]])
    data_dict_tr['tr_seqlen'] = np.squeeze(data_dict_tr['seqlen']).astype(np.int32)
    data_dict_tr['x_tr'], data_dict_tr['y_tr'] = extract_seqs(data_dict_tr['x'], data_dict_tr['y'],
                                                              data_dict_tr['tr_seqlen'], labelled_data_config['tr'])
    np.set_printoptions(threshold=np.nan)

    del data_dict_tr['x']
    del data_dict_tr['y']
    del data_dict_tr['seqlen']
    data_dict_va = loadmat(filenames[labelled_data_config['dataset'][1]])
    data_dict_va['va_seqlen'] = np.squeeze(data_dict_va['seqlen']).astype(np.int32)
    data_dict_va['x_va'], data_dict_va['y_va'] = extract_seqs(data_dict_va['x'], data_dict_va['y'],
                                                              data_dict_va['va_seqlen'], labelled_data_config['va'])
    del data_dict_va['x']
    del data_dict_va['y']
    del data_dict_va['seqlen']
    return {**data_dict_tr, **data_dict_va}
