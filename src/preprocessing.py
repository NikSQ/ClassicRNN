import numpy as np

def extract_sequences(dataset_dict, dataset_config, remove_bias=False):
    in_seq_len = dataset_config["in_seq_len"]
    seqlens = np.squeeze(dataset_dict['seqlen']).astype(np.int32)
    
    x = dataset_dict["x"]
    y = dataset_dict["y"]

    if x.shape[0] != y.shape[0]:
        raise Exception("samples in X != samples in Y")
    
    if remove_bias:
        n_category_samples = []
        category_seq_lens = []
        category_idcs = []

        for category in range(10):
            category_idcs.append(np.where(y[:, category] == 1)[0])
            category_seq_lens.append(seqlens[category_idcs[-1]])
            n_category_samples.append(np.searchsorted(category_seq_lens[-1], in_seq_len))

        n_samples_per_category = np.min(n_category_samples)  # Determines the amount of samples of each category
        selected_idcs = [digit_idc[:n_samples_per_category] for digit_idc in category_idcs]

        x_list = []
        y_list = []
        seqlens_list = []
        for idcs in selected_idcs:
            x_list.append(x[idcs])
            y_list.append(y[idcs])
            seqlens_list.append(seqlens[idcs])

        x_new = np.concatenate(x_list, axis=0)[:, :, :in_seq_len]
        y_new = np.concatenate(y_list, axis=0)
        seqlens_new = np.concatenate(seqlens_list, axis=0)
        return x_new, y_new, seqlens_new

    raise Exception("Not implemented")



def extract_seqs(x, y, seqlens, data_config, remove_bias=False):
    in_seq_len = data_config['in_seq_len']
    seqlens = seqlens.astype(np.int32)

    if x.shape[0] != y.shape[0]:
        raise Exception("samples in X != samples in Y")

    # Iterate over each sample
    n_samples = x.shape[0]
    raw_sample_length = x.shape[2]

    # Iterate over all samples and figure out how many sequences one can extract
    seq_extraction_ranges = []
    for sample_nr in range(n_samples):
        beginning_time_idx = raw_sample_length - seqlens[sample_nr]
        if in_seq_len + data_config['max_truncation'] >= seqlens[sample_nr]:
            ranges = []
            for start_idx in range(beginning_time_idx, beginning_time_idx + seqlens[sample_nr] - in_seq_len):
                ranges.append(range(start_idx, start_idx + in_seq_len))
            if seqlens[sample_nr] <= in_seq_len:
                ranges.append(range(beginning_time_idx, raw_sample_length))
            if len(ranges) > 1:
                seq_extraction_ranges.append(ranges[np.random.choice(np.arange(len(ranges)))])
            else:
                seq_extraction_ranges.append(ranges[0])
        else:
            seq_extraction_ranges.append(range(1, 0))

    n_sequences = sum(len(extraction_range) != 0 for extraction_range in seq_extraction_ranges)
    n_discarded_samples = sum([len(extraction_range) == 0 for extraction_range in seq_extraction_ranges])
    x_shape = (n_sequences, x.shape[1], in_seq_len)
    y_shape = (n_sequences, y.shape[1])

    x_seqs = np.zeros(x_shape)
    y_seqs = np.zeros(y_shape)
    end_time = np.zeros((n_sequences,))

    # Iterate again over the extraction indices and extract the sequences
    seq_idx = 0
    for sample_idx, extraction_range in enumerate(seq_extraction_ranges):
        seq_len = len(extraction_range)
        if len(extraction_range) is not 0:
            x_seqs[seq_idx, :, -seq_len:] = np.transpose(x[sample_idx, :, extraction_range])
            y_seqs[seq_idx] = y[sample_idx, :, -1]
            end_time[seq_idx] = seq_len - 1
            #repeats = int(np.ceil(float(in_seq_len) / float(seq_len)))
            #for repeat in range(1, repeats):
                #if repeat != repeats - 1:
                    #x_seqs[seq_idx, :, repeat * seq_len:(repeat + 1) * seq_len] = x_seqs[seq_idx, :, :seq_len]
                #else:
                    #if in_seq_len % seq_len == 0:
                        #x_seqs[seq_idx, :, repeat * seq_len:] = x_seqs[seq_idx, :, :(seq_len)]
                    #else:
                        #x_seqs[seq_idx, :, repeat * seq_len:] = x_seqs[seq_idx, :, :(in_seq_len % seq_len)]
            seq_idx += 1
    unique, counts = np.unique(np.argmax(y_seqs, axis=1), return_counts=True)

    if remove_bias is False:
        return x_seqs, y_seqs, end_time
    n_samples_per_label = np.min(counts)
    n_tot_samples = n_samples_per_label * len(counts)
    x_shape = (n_tot_samples, x.shape[1], in_seq_len)
    y_shape = (n_tot_samples, y.shape[1])
    new_x_seqs = np.zeros(x_shape)
    new_y_seqs = np.zeros(y_shape)
    new_end_time = np.zeros((n_tot_samples,))
    new_counts = np.zeros_like(counts)
    seq_idx = 0
    for sample_idx in range(n_sequences):
        label = np.argmax(y_seqs[sample_idx])
        if new_counts[label] >= n_samples_per_label:
            continue
        new_counts[label] += 1
        new_x_seqs[seq_idx] = x_seqs[sample_idx]
        new_y_seqs[seq_idx] = y_seqs[sample_idx]
        new_end_time[seq_idx] = end_time[sample_idx]
        seq_idx += 1
    unique, counts = np.unique(np.argmax(new_y_seqs, axis=1), return_counts=True)
    return new_x_seqs, new_y_seqs, new_end_time


