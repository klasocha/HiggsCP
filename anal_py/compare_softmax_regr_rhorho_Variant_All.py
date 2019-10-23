import numpy as np
import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_from_file
from anal_utils import calculate_metrics_regr_c012s_from_file

x_list = list(range(3, 52, 2))
filelist_rhorho_Variant_All = ['../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_3/monit_npy/' for i in x_list]

metrics_softmax_Variant_All = [calculate_metrics_from_file(filelist_rhorho_Variant_All[index], x) for index, x in enumerate(x_list)]
metrics_softmax_Variant_All = np.stack(metrics_softmax_Variant_All)


filelist_rhorho_regr_Variant_All='../laptop_results/nn_rhorho_Variant-All_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/'

metrics_regr_Variant_All = [calculate_metrics_regr_c012s_from_file(filelist_rhorho_regr_Variant_All, x) for x in x_list]
metrics_regr_Variant_All = np.stack(metrics_regr_Variant_All)




# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#binning for horisontal axis
x = np.array(x_list)

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

