import numpy as np
from glob import glob
import os, errno

from sklearn.metrics import roc_auc_score

def calculate_metrics(directory, num_classes):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)
    # ERW including here redefinition of  calc_pred_argmaxs_distances by Michal
    calc_pred_argmaxs_distances = np.min(
       np.stack(
           [np.abs(pred_arg_maxs-calc_arg_maxs), ((num_classes - 1) - np.abs(pred_arg_maxs-calc_arg_maxs))]
       ), axis=0)
    
    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()

    mean_error = np.mean(calc_pred_argmaxs_distances)
    #ERW: scaled to radians and in units of phi^CP
    k2PI = 6.28
    mean_error_scaled = np.mean(calc_pred_argmaxs_distances/(1.0*num_classes) * k2PI/2. )

    # ERW
    # calc_w are not normalised to unity, while preds_w are
    # clarify this point, here l1_delta_w,  l1_delta_w expressed in units of probabilities
    for i in range (len(calc_w)):
      calc_w[i] = calc_w[i]/sum(calc_w[i])
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, mean_error, l1_delta_w, l2_delta_w, mean_error_scaled])

def get_filename_for_class(pathIN, class_num, subset=None):
    d = '../monit_npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_{class_num}'
    if subset:
        d += "_WEIGHTS_SUBS" + str(subset)
    return d

def calculate_roc_auc(pred_w, calc_w, index_a, index_b):
    n, num_classes = calc_w.shape
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([pred_w[:, index_a], pred_w[:, index_a]])
    weights = np.concatenate([calc_w[:, index_a], calc_w[:, index_b]])

    return roc_auc_score(true_labels, preds, sample_weight=weights)

def test_roc_auc(directory, num_class):
    print directory
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))

    roc_auc = []
    oracle_roc_auc = []
    
    for i in range(0, num_class):
         roc_auc += [calculate_roc_auc(preds_w, calc_w, 0, i)]
         oracle_roc_auc += [calculate_roc_auc(calc_w, calc_w, 0, i)]
         print(i, 'roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)),
                  'oracle_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)))
    return oracle_roc_auc,  roc_auc     
