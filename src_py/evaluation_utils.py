""" This module contains different functions responsible for the evaluation
metrics used to measure the performance of the model from tf_model_2.py """

from .tf_model import calculate_deltas_unsigned, calculate_deltas_signed
import numpy as np
import tensorflow as tf

def compute_accuracy_and_mean(model, dataset, batch_size, args, at_most=None, filtered=False):
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
    pred_w = model.predict(x, batch_size=batch_size, verbose=0)
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, n_classes))

    # Computing the mean of the difference between the most probable predicted 
    # class and the most probable true class (∆_class)      
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    mean = np.mean(calculate_deltas_signed(pred_argmaxs, calc_argmaxs, n_classes))

    # ACC (accuracy): averaging that most probable predicted class match for t
    # the most probable class within the ∆_max tolerance. ∆max specifiec the maximum 
    # allowed difference between the predicted class and the true class for an event 
    # to be considered correctly classified.
    delt_max = int(args.DELT_CLASSES)
    acc = (calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, n_classes) <= delt_max).mean()

    # Computing the L1 and L2 norms for the weights  
    l1_delt_w = np.mean(np.abs(calc_w - pred_w))
    l2_delt_w = np.sqrt(np.mean((calc_w - pred_w)**2))
    
    return acc, mean, l1_delt_w, l2_delt_w