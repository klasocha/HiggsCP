import numpy as np
from glob import glob
import os, errno


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
    den = np.abs(pred_arg_maxs-calc_arg_maxs)
    den[den==0]=1
    signed = (pred_arg_maxs-calc_arg_maxs)/den
    calc_pred_argmaxs_distances *= np.int32(signed)
    
    acc0 = (calc_pred_argmaxs_distances <= 0).mean()
    acc1 = (calc_pred_argmaxs_distances <= 1).mean()
    acc2 = (calc_pred_argmaxs_distances <= 2).mean()
    acc3 = (calc_pred_argmaxs_distances <= 3).mean()
    
    meanDelta = np.mean(calc_pred_argmaxs_distances)
    meanDeltaScaled = np.mean(calc_pred_argmaxs_distances/(1.0*num_classes) * 3.14/2. )

    # ERW
    # calc_w are not normalised to unity, while preds_w are
    # clarify this point
    for i in range (len(calc_w)):
      calc_w[i] = calc_w[i]/sum(calc_w[i])
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, meanDelta, l1_delta_w, l2_delta_w, meanDeltaScaled])
