import tensorflow as tf
import numpy as np
from src.logging_tools import get_logger


def extract_seqs(x, y, seqlens, labelled_data_config):
    logger = get_logger('DataContainer')
    in_seq_len = labelled_data_config['in_seq_len']
    out_seq_len = labelled_data_config['out_seq_len']

    if x.shape[0] != y.shape[0]:
        logger.critical("The numbers of samples in X ({}) does not match the number of samples in Y({})"
                        .format(x.shape[0], y.shape[0]))
        raise Exception("samples in X != samples in Y")
    if out_seq_len > in_seq_len:
        logger.critical("The output sequence ({}) can not be longer than the input sequence ({})"
                        .format(out_seq_len, in_seq_len))
        raise Exception("out_seq_len > in_seq_len")

    # Iterate over each sample
    n_samples = x.shape[0]
    raw_sample_length = x.shape[2]
    logger.debug("Shaping {} data samples. InSeqLen: {}, OutSeqLen: {}".format(n_samples, in_seq_len, out_seq_len))

    # Iterate over all samples and figure out how many sequences one can extract
    seq_extraction_ranges = []
    for sample_nr in range(n_samples):
        beginning_time_idx = raw_sample_length - seqlens[sample_nr]

        if labelled_data_config['do_zero_padding']:
            start_idx = max(0, beginning_time_idx - in_seq_len)
        else:
            start_idx = beginning_time_idx

        seq_extraction_ranges.append(range(start_idx, raw_sample_length - in_seq_len))

    n_sequences = sum([len(extraction_range) for extraction_range in seq_extraction_ranges])
    n_discarded_samples = sum([len(extraction_range) == 0 for extraction_range in seq_extraction_ranges])
    x_shape = (n_sequences, x.shape[1], in_seq_len)
    y_shape = (n_sequences, y.shape[1], out_seq_len)
    logger.debug("Discarded {} data samples. Obtained {} sequences".format(n_sequences, n_discarded_samples))

    x_seqs = np.zeros(x_shape)
    y_seqs = np.zeros(y_shape)

    # Iterate again over the extraction indices and extract the sequences
    seq_idx = 0
    for sample_idx, extraction_range in enumerate(seq_extraction_ranges):
        for extraction_idx in extraction_range:
            x_seqs[seq_idx] = x[sample_idx, :, extraction_idx:extraction_idx+in_seq_len]
            y_seqs[seq_idx] = y[sample_idx, :, extraction_idx:extraction_idx+out_seq_len]
            seq_idx += 1

    return x_seqs, y_seqs


class LabelledData:
    def __init__(self, x_shape, y_shape, n_tr_samples, n_va_samples):
        self.n_tr_samples = n_tr_samples
        self.n_va_samples = n_va_samples
        self.x_shape = x_shape
        self.y_shape = y_shape

        with tf.variable_scope('labelled_data'):
            self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,) + x_shape)
            self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,) + y_shape)
            self.x_tr = tf.get_variable(name='x_tr', shape=(n_tr_samples,) + x_shape, dtype=tf.float32)
            self.y_tr = tf.get_variable(name='y_tr', shape=(n_tr_samples,) + y_shape, dtype=tf.float32)
            self.x_va = tf.get_variable(name='x_va', shape=(n_va_samples,) + x_shape, dtype=tf.float32)
            self.y_va = tf.get_variable(name='y_va', shape=(n_va_samples,) + y_shape, dtype=tf.float32)
            self.is_validation = tf.placeholder(name='is_validation', dtype=tf.bool)

            assign_x_tr_op = tf.assign(self.x_tr, self.x_placeholder)
            assign_y_tr_op = tf.assign(self.y_tr, self.y_placeholder)
            self.load_tr_set_op = tf.group(*[assign_x_tr_op, assign_y_tr_op])

            assign_x_va_op = tf.assign(self.x_va, self.x_placeholder)
            assign_y_va_op = tf.assign(self.y_va, self.y_placeholder)
            self.load_va_set_op = tf.group(*[assign_x_va_op, assign_y_va_op])

            self.x = tf.cond(self.is_validation, lambda: self.x_va, lambda: self.x_tr)
            self.y = tf.cond(self.is_validation, lambda: self.y_va, lambda: self.y_tr)






