import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_from_file
from anal_utils import calculate_metrics_regr_popts_from_file

filelist_rhorho_Variant_All = []


filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_2/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_4/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_6/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_8/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_10/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_12/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_14/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_16/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_18/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_20/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_22/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_24/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_26/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_28/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_30/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_32/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_34/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_36/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_38/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_40/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_42/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_44/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_46/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_48/monit_npy/')
filelist_rhorho_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_50/monit_npy/')


metrics_softmax_Variant_All = [calculate_metrics_from_file(filelist_rhorho_Variant_All[1], 4),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[2], 6), calculate_metrics_from_file(filelist_rhorho_Variant_All[3], 8),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[4], 10), calculate_metrics_from_file(filelist_rhorho_Variant_All[5], 12),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[6], 14), calculate_metrics_from_file(filelist_rhorho_Variant_All[7], 16),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[8], 18), calculate_metrics_from_file(filelist_rhorho_Variant_All[9], 20),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[10], 22), calculate_metrics_from_file(filelist_rhorho_Variant_All[11], 24),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[12], 26), calculate_metrics_from_file(filelist_rhorho_Variant_All[13], 28),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[14], 30), calculate_metrics_from_file(filelist_rhorho_Variant_All[15], 32),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[16], 34), calculate_metrics_from_file(filelist_rhorho_Variant_All[17], 36),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[18], 38), calculate_metrics_from_file(filelist_rhorho_Variant_All[19], 40),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[20], 42), calculate_metrics_from_file(filelist_rhorho_Variant_All[21], 44),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[22], 46), calculate_metrics_from_file(filelist_rhorho_Variant_All[23], 48),
                       calculate_metrics_from_file(filelist_rhorho_Variant_All[24], 50)]
           
metrics_softmax_Variant_All = np.stack(metrics_softmax_Variant_All)


filelist_rhorho_regr_Variant_All=[]
filelist_rhorho_regr_Variant_All.append('../laptop_results/nn_rhorho_Variant-All_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/')


metrics_regr_Variant_All = [calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 4),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 6), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 8),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 10), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 12),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 14), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 16),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 18), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 20),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 22), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 24),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 26), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 28),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 30), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 32),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 34), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 36),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 38), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 40),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 42), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 44),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 46), calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 48),
                       calculate_metrics_regr_popts_from_file(filelist_rhorho_regr_Variant_All[0], 50)]
         
metrics_regr_Variant_All = np.stack(metrics_regr_Variant_All)




# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#binning for horisontal axis
x = np.arange(2,26)*2

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_acc_compared_Variant-All_nc"

plt.plot(x, metrics_softmax_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$, multi-class')
plt.plot(x, metrics_softmax_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$, multi-class')
plt.plot(x, metrics_softmax_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$, multi-class')
plt.plot(x, metrics_softmax_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$, multi-class')
plt.plot(x, metrics_regr_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$, regresion')
plt.plot(x, metrics_regr_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$, regresion')
plt.plot(x, metrics_regr_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$, regresion')
plt.plot(x, metrics_regr_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$, regresion')
plt.legend(loc='upper right')
plt.ylim([0.0, 2.2])
plt.xticks(x)
plt.xlabel(r'$N_{class}$')
plt.ylabel('Probability')
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
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()

#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------


pathOUT = "figures/"
filename = "rhorho_L1delt_compared_Variant_All_nc"

plt.plot(x, metrics_softmax_Variant_All[:, 5],'o', label=r'$l_1$ with $wt^{norm}$, multi-class')
plt.plot(x, metrics_regr_Variant_All[:, 12],'d', label=r'$l_1$ with $wt^{norm}$, regression')

plt.ylim([0.0, 0.1])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_1$')
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
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
    
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------



pathOUT = "figures/"
filename = "rhorho_L2delt_compared_Variant_All_nc"

plt.plot(x, metrics_softmax_Variant_All[:, 6],'o', label=r'$l_2$ with $wt^{norm}$, multi-class')
plt.plot(x, metrics_regr_Variant_All[:, 13],'d', label=r'$l_2$ with $wt^{norm}$, regression')

plt.ylim([0.0, 0.1])
plt.xticks(x)
plt.legend()
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$l_2$')
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
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
    
#---------------------------------------------------------------------
plt.clf()
#---------------------------------------------------------------------

