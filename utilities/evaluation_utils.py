""" This module contains different functions responsible for the evaluation
metrics used to measure the performance of the model from tf_model.py """

from .metrics_utils import calculate_deltas_unsigned, calculate_deltas_signed
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def compute_accuracy_and_mean(model, dataset, batch_size, delta_max_tolerance, 
                              at_most=None, filtered=False):
    """ Compute accuracy (within the ∆_max tolerance) and the error mean value """
    x = dataset.x
    calc_w = dataset.weights
    filt = dataset.filt
    
    if at_most:
        x = x[:at_most]
        calc_w = calc_w[:at_most]
        filt = filt[:at_most]
    if filtered:
        x = x[filt == 1.0]
        calc_w = calc_w[filt == 1.0]
    
    n_classes = calc_w.shape[-1]
    pred_w = model.predict(x, batch_size, verbose=0)
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, n_classes))

    # Computing the mean of the difference between the most probable predicted 
    # class and the most probable true class (∆_class)      
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    mean = np.mean(calculate_deltas_signed(calc_argmaxs, pred_argmaxs, n_classes))

    # ACC (accuracy): averaging that most probable predicted class match for t
    # the most probable class within the ∆_max tolerance. ∆max specifiec the maximum 
    # allowed difference between the predicted class and the true class for an event 
    # to be considered correctly classified.
    acc = (calculate_deltas_unsigned(calc_argmaxs, pred_argmaxs, n_classes) 
           <= delta_max_tolerance).mean()

    # Computing the L1 and L2 norms for the weights  
    l1_delt_w = np.mean(np.abs(calc_w - pred_w))
    l2_delt_w = np.sqrt(np.mean((calc_w - pred_w)**2))
    
    return acc, mean, l1_delt_w, l2_delt_w


def compute_loss(model, dataset, batch_size, filtered=False):
    n_epochs = dataset.n // batch_size
    losses = []
    for _ in range(n_epochs):
        x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s, filt = dataset.next_batch(batch_size)
        if model.configuration == "soft_weights":
            labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), 
                                    (1, weights.shape[-1]))
        if model.configuration == "soft_argmaxs":
            hits_argmaxs = hits_argmaxs[:, :-1]
            labels = hits_argmaxs / tf.tile(tf.reshape(tf.reduce_sum(hits_argmaxs, axis=1), 
                                                        (-1, 1)), (1, hits_argmaxs.shape[-1]))
        if model.configuration == "soft_c012s":
            labels = hits_c012s / tf.tile(tf.reshape(tf.reduce_sum(hits_c012s, axis=1), 
                                                        (-1, 1)), (1, hits_c012s.shape[-1]))
        if model.configuration == "regr_argmaxs":
            labels = argmaxs
        if model.configuration == "regr_c012s":
            labels = c012s
        if model.configuration == "regr_weights":
            labels = weights
        if filtered:
            x = x[filt == 1.0]
            labels = labels[filt == 1.0]
        losses.append(model.loss(labels, model.predict_on_batch(x)))
    return np.mean(losses)


def calculate_roc_auc(pred_w, calc_w, index_a, index_b):
    """ Calculate the ROC for a specific pair of classes (useful for multiclass classification).
    This function is used by test_roc_auc() """
    n, _ = calc_w.shape
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([pred_w[:, index_a], pred_w[:, index_a]])
    weights = np.concatenate([calc_w[:, index_a], calc_w[:, index_b]])
    return roc_auc_score(true_labels, preds, sample_weight=weights)


def test_roc_auc(preds_w, calc_w):
    """ Test the ROC AUC for each class. This function calculates and prints the ROC AUC 
    for each class based on the predicted weights and the calculated weights. """
    n, num_classes = calc_w.shape
    for i in range(0, num_classes):
         print(i + 1, 'oracle_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)),
                  'roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)))