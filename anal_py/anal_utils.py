import numpy as np
from glob import glob
import os, errno

from sklearn.metrics import roc_auc_score

from src_py.metrics_utils import calculate_deltas_unsigned, calculate_deltas_signed


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

def calc_weights(num_classes, popts):
    x = np.linspace(0, 2, num_classes) * np.pi
    data_len = popts.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *popts[i])
    return weights

def calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_class):
    return calculate_deltas_signed(calc_arg_maxs, pred_arg_maxs, num_class)


def calculate_metrics_from_file(directory, num_classes):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    return calculate_metrics(calc_w, preds_w, num_classes)


def calculate_metrics(calc_w, preds_w, num_classes):
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)

    calc_pred_argmaxs_abs_distances = calculate_deltas_unsigned(calc_arg_maxs, pred_arg_maxs, num_classes)
    calc_pred_argmaxs_signed_distances = calculate_deltas_signed(calc_arg_maxs, pred_arg_maxs, num_classes)
    k2PI = 6.28
    calc_pred_argmaxs_abs_distances_rad = calc_pred_argmaxs_abs_distances * k2PI/(1.0*num_classes)
    
    print "abs_distances", calc_pred_argmaxs_abs_distances
    print "abs_distances[rad]", calc_pred_argmaxs_abs_distances_rad
    print "signed_distances", calc_pred_argmaxs_signed_distances
    print "mean distance", np.mean(calc_pred_argmaxs_signed_distances), np.mean(calc_pred_argmaxs_signed_distances) * 360./(1.0*num_classes),"[deg]"

    mean_deltas = np.mean(calc_pred_argmaxs_signed_distances)
    #ERW: scaled to radians and in units of alpha^CP
    mean_deltas_rad = mean_deltas * k2PI/(1.0*num_classes)
    print mean_deltas_rad

    acc0 = (calc_pred_argmaxs_abs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_abs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_abs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_abs_distances <= 3).mean()

    print "fraction in 0, 1, 2, 3 class", acc0, acc1, acc2, acc3

    acc0_rad = (calc_pred_argmaxs_abs_distances_rad <= 0.25).mean()
    acc1_rad = (calc_pred_argmaxs_abs_distances_rad <= 0.50).mean()
    acc2_rad = (calc_pred_argmaxs_abs_distances_rad <= 1.75).mean()
    acc3_rad = (calc_pred_argmaxs_abs_distances_rad <= 1.0).mean()

    print "fraction in 0.25, 0.50, 0.75, 1.0 [rad]", acc0_rad, acc1_rad, acc2_rad, acc3_rad

    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, mean_deltas, l1_delta_w, l2_delta_w, mean_deltas_rad, acc0_rad, acc1_rad, acc2_rad, acc3_rad])


def calculate_metrics_regr_popts_from_file(directory, num_classes):
    calc_popts = np.load(os.path.join(directory,'test_regr_calc_popts.npy'))
    pred_popts = np.load(os.path.join(directory,'test_regr_preds_popts.npy'))

    return calculate_metrics_regr_popts(calc_popts, pred_popts, num_classes)


def calculate_metrics_regr_popts(calc_popts, pred_popts, num_classes):
    calc_w  = calc_weights(num_classes, calc_popts)
    preds_w = calc_weights(num_classes, pred_popts)

    return calculate_metrics(calc_w, preds_w, num_classes)


def get_filename_for_class(pathIN, class_num, subset=None):
    d = '../monit_npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_{class_num}'
    if subset:
        d += "_WEIGHTS_SUBS" + str(subset)
    return d


# The primary versions of three methods below were
#  evaluate from tf_model.py
#  evaluate2 from tf_model.py
#  both using 
#  evaluate_preds  from tf_model.py
# when extending to multi-class something is not correctly
# implemented for handling numpy arrays. 


def evaluate_roc_auc(preds, wa, wb):
    n = len(preds)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds, preds])
    weights = np.concatenate([wa, wb])
    
    return roc_auc_score(true_labels, preds, sample_weight=weights)


def calculate_roc_auc(preds_w, calc_w, index_a, index_b):
    n, num_classes = calc_w.shape
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds_w[:, index_a], preds_w[:, index_a]])
    weights = np.concatenate([calc_w[:, index_a], calc_w[:, index_b]])

    return roc_auc_score(true_labels, preds, sample_weight=weights)

# binary classification
def test_roc_auc(directory, num_class):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))
    
    oracle_roc_auc = []
    preds_roc_auc  = []
    
    for i in range(0, num_class):
         oracle_roc_auc  += [calculate_roc_auc(calc_w, calc_w, 0, i)]
         preds_roc_auc   += [calculate_roc_auc(preds_w, calc_w, 0, i)]
         print(i,
                  'oracle_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)),
                  'preds_roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)))

    return oracle_roc_auc, preds_roc_auc
