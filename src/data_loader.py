from scipy.io import loadmat
import numpy as np
from src.data_tools import extract_seqs


filenames = {'pen_stroke_small': '../datasets/mnist_pen_strokes/mnist_pen_stroke_5000_1000.mat'}


def load_dataset(labelled_data_config):
    data_dict = loadmat(filenames[labelled_data_config['dataset']])
    data_dict['tr_seqlen'] = np.squeeze(data_dict['tr_seqlen']).astype(np.int32)
    data_dict['va_seqlen'] = np.squeeze(data_dict['va_seqlen']).astype(np.int32)
    data_dict['x_tr'], data_dict['y_tr'] = extract_seqs(data_dict['x_tr'], data_dict['y_tr'], data_dict['tr_seqlen'],
                                                        labelled_data_config)
    data_dict['x_va'], data_dict['y_va'] = extract_seqs(data_dict['x_va'], data_dict['y_va'], data_dict['va_seqlen'],
                                                        labelled_data_config)
    return data_dict
