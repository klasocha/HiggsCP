import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt


filelist_a1rho_Variant_All = []

filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_2')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_6')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_8')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_12')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_14')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_16')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_18')
filelist_a1rho_Variant_All.append('npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_20')

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
    
    meanDelta = np.mean(calc_pred_argmaxs_distances)
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w)**2))
    
    return np.array([acc0, acc1, acc2, acc3, meanDelta, l1_delta_w, l2_delta_w])



metrics_Variant_All = [calculate_metrics(filelist_a1rho_Variant_All[0], 2),  calculate_metrics(filelist_a1rho_Variant_All[1], 4),
           calculate_metrics(filelist_a1rho_Variant_All[1], 6), calculate_metrics(filelist_a1rho_Variant_All[2], 8),
           calculate_metrics(filelist_a1rho_Variant_All[3], 10), calculate_metrics(filelist_a1rho_Variant_All[3], 12),
           calculate_metrics(filelist_a1rho_Variant_All[4], 14), calculate_metrics(filelist_a1rho_Variant_All[4], 16),
           calculate_metrics(filelist_a1rho_Variant_All[4], 18), calculate_metrics(filelist_a1rho_Variant_All[4], 20)]
           
metrics_Variant_All = np.stack(metrics_Variant_All)


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "a1rho_Variant_All_acc_nc"
x = np.arange(1,11)*2
# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 0],'o', label='Acc0')
plt.plot(x, metrics_Variant_All[:, 1],'x', label='Acc1')
plt.plot(x, metrics_Variant_All[:, 2],'d', label='Acc2')
plt.plot(x, metrics_Variant_All[:, 3],'v', label='Acc3')
plt.ylim([0.0, 1.3])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel('Accuracy')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.show()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "a1rho_Variant_All_meanDelta_nc"

plt.plot(x, metrics_Variant_All[:, 4],'o', label=r'$<\Delta>$ classes')

plt.ylim([0.0, 3])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta>$ classes')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.show()
#---------------------------------------------------------------------


pathOUT = "figures/"
filename = "a1rho_Variant_All_L1delt_nc"

plt.plot(x, metrics_Variant_All[:, 5],'o', label=r'L1 $<\Delta w>$')

plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'L1 $<\Delta w>$')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
else:
    plt.show()
    
#---------------------------------------------------------------------
plt.show()
#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "a1rho_Variant_All_L2delt_nc"

plt.plot(x, metrics_Variant_All[:, 6],'o', label=r'L2 $<\Delta w>$')

plt.ylim([0.0, 2.0])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'L2 $<\Delta w>$')
plt.title('Feautures list: Variant-All')

ax = plt.gca()
plt.tight_layout()

if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
else:
    plt.show()
#---------------------------------------------------------------------
plt.show()
#---------------------------------------------------------------------
 