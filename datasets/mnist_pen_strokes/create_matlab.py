from scipy.io import savemat
import numpy as np

# custom config
n_tr_samples = 5000
n_va_samples = 1000
file_name = 'mnist_pen_stroke_5000_1000.mat'

# everything else
n_tot_tr_samples = 60000
n_tot_va_samples = 10000

sel_tr_samples = np.random.choice(np.arange(n_tot_tr_samples), size=(n_tr_samples), replace=False)
sel_va_samples = np.random.choice(np.arange(n_tot_va_samples), size=(n_va_samples), replace=False)

tr_inputs = []
tr_targets = []
tr_lengths = []
max_len = 0
for sample_nr in sel_tr_samples:
  tr_input = np.loadtxt('download/trainimg-' + str(sample_nr) + '-inputdata.txt')
  tr_target = np.loadtxt('download/trainimg-' + str(sample_nr) + '-targetdata.txt')
  tr_lengths.append(tr_input.shape[0] - 1)
  if max_len < tr_input.shape[0] - 1:
    max_len = tr_input.shape[0] - 1
  tr_inputs.append(tr_input[1:, :])
  tr_targets.append(tr_target[1:, :])

va_inputs = []
va_targets =[]
va_lengths = []
for sample_nr in sel_va_samples:
  va_input = np.loadtxt('download/testimg-' + str(sample_nr) + '-inputdata.txt')
  va_target = np.loadtxt('download/testimg-' + str(sample_nr) + '-targetdata.txt')
  va_lengths.append(va_input.shape[0] - 1)
  if max_len < va_input.shape[0] - 1:
    max_len = va_input.shape[0] - 1
  va_inputs.append(va_input[1:, :])
  va_targets.append(va_target[1:, :])

tr_inputs_arr = np.zeros((len(tr_inputs), 4, max_len), dtype=float)
tr_pred_targets_arr = np.zeros((len(tr_targets), 4, max_len), dtype=float)
tr_class_targets_arr = np.zeros((len(tr_targets), 10, max_len), dtype=float)
tr_idc = np.argsort(tr_lengths)
tr_seqlen = np.zeros((len(tr_inputs),))
for i in range(len(tr_inputs)):
  idx = np.where(tr_idc == i)[0][0]
  tr_inputs_arr[idx, :, (max_len - tr_inputs[i].shape[0]):] = np.transpose(tr_inputs[i])
  tr_seqlen[idx] = tr_inputs[i].shape[0]
  tr_pred_targets_arr[idx, :, (max_len - tr_targets[i].shape[0]):] = np.transpose(tr_targets[i])[10:, :]
  tr_class_targets_arr[idx, :, (max_len - tr_targets[i].shape[0]):] = np.transpose(tr_targets[i])[:10, :]

va_inputs_arr = np.zeros((len(va_inputs), 4, max_len), dtype=float)
va_pred_targets_arr = np.zeros((len(va_targets), 4, max_len), dtype=float)
va_class_targets_arr = np.zeros((len(va_targets), 10, max_len), dtype=float)
va_idc = np.argsort(va_lengths)
va_seqlen = np.zeros((len(va_inputs),))
for i in range(len(va_inputs)):
  idx = np.where(va_idc == i)[0][0]
  va_inputs_arr[idx, :, (max_len - va_inputs[i].shape[0]):] = np.transpose(va_inputs[i])
  va_seqlen[idx] = va_inputs[i].shape[0]
  va_pred_targets_arr[idx, :, (max_len - va_targets[i].shape[0]):] = np.transpose(va_targets[i])[10:, :]
  va_class_targets_arr[idx, :, (max_len - va_targets[i].shape[0]):] = np.transpose(va_targets[i])[:10, :]

savemat(file_name, mdict={'x_tr':tr_inputs_arr, 'y_tr':tr_class_targets_arr, 'x_va':va_inputs_arr, 'y_va':va_class_targets_arr, 'tr_seqlen':tr_seqlen, 'va_seqlen':va_seqlen})

