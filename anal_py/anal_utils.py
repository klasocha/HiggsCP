import numpy as np
from glob import glob
import os, errno

from sklearn.metrics import roc_auc_score


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

def calc_weights(num_classes, popts):
    x = np.linspace(0, 2, num_classes) * np.pi
    data_len = popts.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *popts[i])
    return weights

# definition from Michal
# something wrong with it, gives oscilating values
def calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_class):
    min_distances = np.zeros(len(calc_arg_maxs))
    for i in range(len(calc_arg_maxs)):
        dist = pred_arg_maxs[i] - calc_arg_maxs[i]
        if np.abs((num_class - 1) + (pred_arg_maxs[i] - calc_arg_maxs[i]))<np.abs(dist):
            dist =(num_class - 1) + (pred_arg_maxs[i] - calc_arg_maxs[i])
        if np.abs(-(num_class - 1) + (pred_arg_maxs[i] - calc_arg_maxs[i]))<np.abs(dist):
            dist = -(num_class - 1) + (pred_arg_maxs[i] - calc_arg_maxs[i])
        min_distances[i]  = dist
    return min_distances

def calculate_metrics(directory, num_classes):
    calc_w  = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))    
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)
    
    # ERW including here redefinition of  calc_pred_argmaxs_distances by Michal
    # short cut is taken in the "sign", should not take mean of absolute value!!!
    # example implementation from anal_py/plot_softmax_rhorho_Variant_All.py

    # delt_argmax = np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)
    # for i in range (len(delt_argmax)):
    # if  delt_argmax[i] > num_classes/2.0 :
    #     delt_argmax[i] = num_classes -  delt_argmax[i]
    # if  delt_argmax[i] < - num_classes/2.0 :
    #    delt_argmax[i] = - num_classes -  delt_argmax[i]
    
    calc_pred_argmaxs_distances = np.min(
         np.stack(
            [np.abs(pred_arg_maxs-calc_arg_maxs), ((num_classes - 1) - np.abs(pred_arg_maxs-calc_arg_maxs))]
        ), axis=0)

    print calc_pred_argmaxs_distances
    print np.mean(calc_pred_argmaxs_distances), np.mean(calc_pred_argmaxs_distances) * 360./(1.0*num_classes)

    mean_error = np.mean(calc_pred_argmaxs_distances)
    #ERW: scaled to radians and in units of alpha^CP
    k2PI = 6.28
    mean_error_scaled = np.mean(calc_pred_argmaxs_distances) * k2PI/(1.0*num_classes)
    print mean_error_scaled
    
    # new definition from Michal
    # something qrong with this definition, oscilating values
    # calc_pred_argmaxs_distances = calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_class)

    # here it is OK, because we are interested in fractions
    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()

    # ERW
    # calc_w are not normalised to unity, while preds_w are
    # clarify this point, here l1_delta_w,  l1_delta_w expressed in units of probabilities
    for i in range (len(calc_w)):
      calc_w[i] = calc_w[i]/sum(calc_w[i])
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, mean_error, l1_delta_w, l2_delta_w, mean_error_scaled])

def calculate_metrics_regr_popts(directory, num_classes):

    calc_popts = np.load(os.path.join(directory,'test_regr_calc_popts.npy'))
    pred_popts = np.load(os.path.join(directory,'test_regr_preds_popts.npy'))
    calc_w  = calc_weights(num_classes, calc_popts)
    preds_w = calc_weights(num_classes, pred_popts)

    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)
    
    # ERW including here redefinition of  calc_pred_argmaxs_distances by Michal
    calc_pred_argmaxs_distances = np.min(
         np.stack(
            [np.abs(pred_arg_maxs-calc_arg_maxs), ((num_classes - 1) - np.abs(pred_arg_maxs-calc_arg_maxs))]
         ), axis=0)

    # new definition from Michal
    # something wrong with this definition, oscilating values
    # calc_pred_argmaxs_distances = calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_classes)
    
    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()

    mean_error = np.mean(calc_pred_argmaxs_distances)
    #ERW: scaled to radians and in units of alpha^CP
    k2PI = 6.28
    mean_error_scaled = np.mean(calc_pred_argmaxs_distances/(1.0*num_classes) * k2PI/2. )

    # ERW
    # calc_w are not normalised to unity, while preds_w are
    # clarify this point, here l1_delta_w,  l1_delta_w expressed in units of probabilities
    for i in range (len(calc_w)):
      calc_w[i]  = calc_w[i]/sum(calc_w[i])
      preds_w[i] = preds_w[i]/sum(preds_w[i])
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, mean_error, l1_delta_w, l2_delta_w, mean_error_scaled])

def get_filename_for_class(pathIN, class_num, subset=None):
    d = '../monit_npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_{class_num}'
    if subset:
        d += "_WEIGHTS_SUBS" + str(subset)
    return d


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
