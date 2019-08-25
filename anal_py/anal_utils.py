import os

import numpy as np
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

    calc_pred_argmaxs_distances = calculate_deltas_unsigned(calc_arg_maxs, pred_arg_maxs, num_classes)

    # new definition from Michal
    # something qrong with this definition, oscilating values
    # calc_pred_argmaxs_distances = calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_class)

    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()

    mean_error = np.mean(calculate_deltas_signed(calc_arg_maxs, pred_arg_maxs, num_classes))
    # ERW: scaled to radians and in units of phi^CP
    k2PI = 6.28
    mean_error_scaled = mean_error / (1.0 * num_classes) * k2PI / 2.

    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w) ** 2))

    return np.array([acc0, acc1, acc2, acc3, mean_error, l1_delta_w, l2_delta_w, mean_error_scaled])


def calculate_metrics_regr_popts(directory, num_classes):
    calc_popts = np.load(os.path.join(directory, 'test_regr_calc_popts.npy'))
    pred_popts = np.load(os.path.join(directory, 'test_regr_preds_popts.npy'))
    calc_w = calc_weights(num_classes, calc_popts)
    preds_w = calc_weights(num_classes, pred_popts)

    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)

    # ERW including here redefinition of  calc_pred_argmaxs_distances by Michal
    calc_pred_argmaxs_distances = calculate_deltas_unsigned(calc_arg_maxs, pred_arg_maxs, num_classes)

    # new definition from Michal
    # something wrong with this definition, oscilating values
    # calc_pred_argmaxs_distances = calc_argmaxs_distances(pred_arg_maxs, calc_arg_maxs, num_classes)

    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()

    mean_error = np.mean(calc_pred_argmaxs_distances)
    # ERW: scaled to radians and in units of phi^CP
    k2PI = 6.28
    mean_error_scaled = np.mean(calc_pred_argmaxs_distances / (1.0 * num_classes) * k2PI / 2.)

    # ERW
    # calc_w are not normalised to unity, while preds_w are
    # clarify this point, here l1_delta_w,  l1_delta_w expressed in units of probabilities
    for i in range(len(calc_w)):
        calc_w[i] = calc_w[i] / sum(calc_w[i])
        preds_w[i] = preds_w[i] / sum(preds_w[i])
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w) ** 2))

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


def test_roc_auc(directory, num_class):
    calc_w = np.load(os.path.join(directory, 'softmax_calc_w.npy'))
    preds_w = np.load(os.path.join(directory, 'softmax_preds_w.npy'))

    calc_roc_auc = []
    preds_roc_auc = []

    # ERW: oracle part is not corrrectly calculated.
    # I don't understand method well enough. Need clarifiction

    for i in range(0, num_class):
        #         oracle_roc_auc += [evaluate_roc_auc(calc_w[0]/(calc_w[0]+calc_w[i]), calc_w[0],  calc_w[i])]
        preds_roc_auc += [calculate_roc_auc(preds_w, calc_w, 0, i)]
        calc_roc_auc += [calculate_roc_auc(calc_w, calc_w, 0, i)]
        print(i,
              #                 'oracle_roc_auc: {}'.format(evaluate_roc_auc(calc_w[0]/(calc_w[0]+calc_w[i]), calc_w[0],  calc_w[i])),
              'preds_roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)),
              'calc_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)))

    return calc_roc_auc, preds_roc_auc
