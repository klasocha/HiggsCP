import numpy as np
from glob import glob
import os

import matplotlib.pyplot as plt


filelist_rhorho_Variant_All = []

filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_2')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_6')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_8')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_12')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_14')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_16')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_18')
filelist_rhorho_Variant_All.append('npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_20')

filelist_rhorho_Variant_4_1 = []

filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_2')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_4')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_6')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_8')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_10')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_12')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_14')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_16')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_18')
filelist_rhorho_Variant_4_1.append('npy/nn_rhorho_Variant-4.1_Unweighted_False_NO_NUM_CLASSES_20')

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



metrics_Variant_All = [calculate_metrics(filelist_rhorho_Variant_All[0], 2),  calculate_metrics(filelist_rhorho_Variant_All[1], 4),
           calculate_metrics(filelist_rhorho_Variant_All[1], 6), calculate_metrics(filelist_rhorho_Variant_All[2], 8),
           calculate_metrics(filelist_rhorho_Variant_All[3], 10), calculate_metrics(filelist_rhorho_Variant_All[3], 12),
           calculate_metrics(filelist_rhorho_Variant_All[4], 14), calculate_metrics(filelist_rhorho_Variant_All[4], 16),
           calculate_metrics(filelist_rhorho_Variant_All[4], 18), calculate_metrics(filelist_rhorho_Variant_All[4], 20)]
           
metrics_Variant_All = np.stack(metrics_Variant_All)


metrics_Variant_4_1 = [calculate_metrics(filelist_rhorho_Variant_4_1[0], 2),  calculate_metrics(filelist_rhorho_Variant_4_1[1], 4),
           calculate_metrics(filelist_rhorho_Variant_4_1[1], 6), calculate_metrics(filelist_rhorho_Variant_4_1[2], 8),
           calculate_metrics(filelist_rhorho_Variant_4_1[3], 10), calculate_metrics(filelist_rhorho_Variant_4_1[3], 12),
           calculate_metrics(filelist_rhorho_Variant_4_1[4], 14), calculate_metrics(filelist_rhorho_Variant_4_1[4], 16),
           calculate_metrics(filelist_rhorho_Variant_4_1[4], 18), calculate_metrics(filelist_rhorho_Variant_4_1[4], 20)]
           
metrics_Variant_4_1 = np.stack(metrics_Variant_4_1)


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps files

plt.plot(metrics_Variant_All[:, 0])
plt.plot(metrics_Variant_All[:, 1])
plt.plot(metrics_Variant_All[:, 2])
plt.plot(metrics_Variant_All[:, 3])
plt.show()
plt.plot(metrics_Variant_All[:, 4])
plt.show()
plt.plot(metrics_Variant_All[:, 5])
plt.show()
plt.plot(metrics_Variant_All[:, 6])
plt.show()

plt.plot(metrics_Variant_4_1[:, 0])
plt.plot(metrics_Variant_4_1[:, 1])
plt.plot(metrics_Variant_4_1[:, 2])
plt.plot(metrics_Variant_4_1[:, 3])
plt.show()
plt.plot(metrics_Variant_4_1[:, 4])
plt.show()
plt.plot(metrics_Variant_4_1[:, 5])
plt.show()
plt.plot(metrics_Variant_4_1[:, 6])
plt.show()
