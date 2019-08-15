import numpy as np
import pprint
import tensorflow as tf


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


def process_results(result_config, result_dicts):
    tr_accs, tr_losses = convert_to_array(result_dicts, 'tr')
    va_accs, va_losses = convert_to_array(result_dicts, 'va')
    te_accs, te_losses = convert_to_array(result_dicts, 'te')

    if result_config['save_results']:
        np.save(result_config['filename'] + '_tr_accs', tr_accs)
        np.save(result_config['filename'] + '_tr_losses', tr_losses)
        np.save(result_config['filename'] + '_va_accs', va_accs)
        np.save(result_config['filename'] + '_va_losses', va_losses)
        np.save(result_config['filename'] + '_te_accs', te_accs)
        np.save(result_config['filename'] + '_te_losses', te_losses)

    #if result_config['plot_results']:
        #plt.plot(tr_epochs, np.mean(tr_accs, axis=0))
        #plt.plot(va_epochs, np.mean(va_accs, axis=0))
        #plt.legend(['training', 'validation'])
        #plt.xlabel('epoch')
        #plt.ylabel('accuracy')
        #plt.show()

    if result_config['print_final_stats']:
        print_stats('Final Tr Acc', tr_accs[:, -1])
        print_stats('Final Va Acc', va_accs[:, -1])
        print_stats('Final Tr Loss', tr_losses[:, -1])
        print_stats('Final Va Loss', va_losses[:, -1])
        print_best('TrAcc', tr_accs)
        epochs = print_best('VaAcc', va_accs)
        print_best('TeAcc', te_accs, epochs)
        print(tr_accs.shape)


def convert_to_array(result_dicts, dict_key):
    accs = []
    losses = []
    for run, result_dict in enumerate(result_dicts):
        accs.append(np.expand_dims(np.asarray(result_dict[dict_key]['accs'], np.float32), axis=0))
        losses.append(np.expand_dims(np.asarray(result_dict[dict_key]['loss'], np.float32), axis=0))
    accs = np.concatenate(accs)
    losses = np.concatenate(losses)
    return accs, losses


def print_stats(name, values):
    print('{:15s}: {:9.5f} +- {:7.5f}'.format(name, np.mean(values), np.std(values, ddof=1)))


def print_best(name, values, epochs=None):
    print('')
    print(name)
    early_stop = True
    if epochs is None:
        epochs = []
        early_stop = False
    for run in range(values.shape[0]):
        if early_stop is False:
            epochs.append(np.argmax(values[run, :]))
        print('Run: {:2d} | {:4.2f}% in epoch {:4d}'.format(run, 100*values[run, epochs[run]], epochs[run]))
    return epochs


def print_config(training_config, data_config, rnn_config):
    print('\n=============================\nCONFIG FILE')
    print('\nRNN CONFIG')
    pprint.pprint(training_config)
    print('\nDATA CONFIG')
    pprint.pprint(data_config)
    print('\nTRAINING CONFIG')
    pprint.pprint(rnn_config)
    print('==============================\n\n')


momentum = None

def set_momentum(value):
    global momentum
    momentum = value

def get_batchnormalizer():
    gamma_init = tf.constant_initializer(value=.1)
    return tf.keras.layers.BatchNormalization(axis=1, center=False, gamma_initializer=gamma_init, momentum=momentum)




