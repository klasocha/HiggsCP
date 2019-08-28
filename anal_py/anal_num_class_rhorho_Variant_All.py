import numpy as np
from glob import glob
import os, errno

import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_from_file

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


metrics_Variant_All = [calculate_metrics_from_file(filelist_rhorho_Variant_All[1], 4),
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
           
metrics_Variant_All = np.stack(metrics_Variant_All)

#binning for vertical axis
x = np.arange(2,26)*2


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_acc_Variant-All_nc"
# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$|\Delta_{class}| < 1$')
plt.plot(x, metrics_Variant_All[:, 1],'x', label=r'$|\Delta_{class}| < 2$')
plt.plot(x, metrics_Variant_All[:, 2],'d', label=r'$|\Delta_{class}| < 3$')
plt.plot(x, metrics_Variant_All[:, 3],'v', label=r'$|\Delta_{class}| < 4$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
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
filename = "rhorho_acc_alphaCP_Variant-All_nc"

# example plt.plot(x, metrics_Variant_All[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_All[:, 8],'o', label=r'$|\Delta\alpha^{CP}| < 0.25[rad]$')
plt.plot(x, metrics_Variant_All[:, 9],'x', label=r'$|\Delta\alpha^{CP}| < 0.50[rad]$')
plt.plot(x, metrics_Variant_All[:, 10],'d', label=r'$|\Delta\alpha^{CP}| < 0.75[rad]$')
plt.plot(x, metrics_Variant_All[:, 11],'v', label=r'$|\Delta\alpha^{CP}| < 1.0[rad]$')
plt.ylim([0.0, 1.5])
plt.xticks(x)
plt.legend()
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
filename = "rhorho_meanDelt_class_Variant-All_nc"

plt.plot(x, metrics_Variant_All[:, 4],'o', label=r'$<\Delta_{class}> [idx]$')

plt.ylim([-0.5, 0.5])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta_{class}>$')
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
filename = "rhorho_meanDelt_alphaCP_Variant-All_nc"

plt.plot(x, metrics_Variant_All[:, 7],'o', label=r'$<\Delta \alpha^{CP}> [rad]$ ')

#plt.ylim([0.0, 0.5])
plt.xticks(x)
plt.legend()
#plt.ylim([-0.5, 0.5])
plt.ylim([-0.3, 0.3])
plt.xlabel(r'$N_{class}$')
plt.ylabel(r'$<\Delta \alpha^{CP}>$')
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
filename = "rhorho_L1delt_w_Variant_All_nc"

plt.plot(x, metrics_Variant_All[:, 12],'o', label=r'$l_1$ with $wt^{norm}$')

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
filename = "rhorho_L2delt_w_Variant_All_nc"

plt.plot(x, metrics_Variant_All[:, 13],'o', label=r'$l_2$ with $wt^{norm}$')

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
 
