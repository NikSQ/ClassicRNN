import numpy as np
from src.logging_tools import get_logger

def extract_seqs(x, y, seqlens, data_config):
    logger = get_logger('DataContainer')
    in_seq_len = data_config['in_seq_len']
    out_seq_len = data_config['out_seq_len']

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

            if data_config['extract_seqs']:
                if data_config['zero_padding'] > 0:
                    start_idx = max(0, beginning_time_idx - min(in_seq_len, data_config['zero_padding']))
                else:
                    start_idx = beginning_time_idx
                seq_extraction_ranges.append(range(start_idx, raw_sample_length - in_seq_len))
            elif in_seq_len >= seqlens[sample_nr]:
                seq_extraction_ranges.append(range(raw_sample_length - in_seq_len,
                                                   raw_sample_length - in_seq_len + 1))
            else:
                seq_extraction_ranges.append(range(1, 0))

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
            y_seqs[seq_idx] = y[sample_idx, :, extraction_idx+in_seq_len-out_seq_len:extraction_idx+in_seq_len]
            seq_idx += 1
    return x_seqs, y_seqs


