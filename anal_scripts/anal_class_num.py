import numpy as np
from glob import glob
import os

import matplotlib.pyplot as plt


filenames = []

filenames.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_2')
filenames.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4')
filenames.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_8')
filenames.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10')
filenames.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_16')

def calculate_metrics(directory, num_class):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)
    calc_pred_argmaxs_distances = np.min(
       np.stack(
           [np.abs(pred_arg_maxs-calc_arg_maxs), (num_class - np.abs(pred_arg_maxs-calc_arg_maxs))]
       ), axis=0)
    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()
    
    mse = np.mean(calc_pred_argmaxs_distances)
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, mse, l1_delta_w, l2_delta_w])



metrics = [calculate_metrics(filenames[0], 2), calculate_metrics(filenames[1], 4),
           calculate_metrics(filenames[2], 8), calculate_metrics(filenames[3], 10),
           calculate_metrics(filenames[4], 16)]

metrics = np.stack(metrics)

plt.plot(metrics[:, 0])
plt.plot(metrics[:, 1])
plt.plot(metrics[:, 2])
plt.plot(metrics[:, 3])
plt.show()
plt.plot(metrics[:, 4])
plt.show()
