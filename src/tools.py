import numpy as np


def generate_init_values(init_config, w_shape, b_shape):
    if init_config['w'] == 'xavier':
        w_init_vals = np.random.randn(w_shape[0], w_shape[1]) * np.sqrt(2/(w_shape[0] + w_shape[1]))
    elif init_config['w'] == 'all_zero':
        w_init_vals = np.zeros(w_shape)
    else:
        raise Exception("{} is not a valid weight initialization".format(init_config['w']))

    if init_config['b'] == 'all_zero':
        b_init_vals = np.zeros(b_shape)
    elif init_config['b'] == 'all_one':
        b_init_vals = np.ones(b_shape)
    else:
        raise Exception("{} is not a valid bias initialization".format(init_config['b']))

    return w_init_vals, b_init_vals

