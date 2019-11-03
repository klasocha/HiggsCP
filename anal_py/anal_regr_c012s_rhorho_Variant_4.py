import numpy as np
import os, errno

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from anal_utils import calculate_metrics_regr_c012s, save_plot_file

filelist_rhorho_Variant_1 = 'npy/nn_rhorho_Variant-4.1_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/'

x_list = list(range(3, 22, 2))

metrics_Variant_1 = [calculate_metrics_regr_c012s(filelist_rhorho_Variant_1, i) for i in x_list]
metrics_Variant_1 = np.stack(metrics_Variant_1)


# Now start plotting metrics
# Make better plots here, add axes labels, add color labels, store into figures/*.eps, figures/*.pdf files
# Should we first convert to histograms (?)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_acc_Variant-4.1_regr"
x = np.arange(1,11)*2+1
# example plt.plot(x, metrics_Variant_1[:, 0],'o', label=r'$\sigma$' )
plt.plot(x, metrics_Variant_1[:, 0],'o', label='Acc0')
plt.plot(x, metrics_Variant_1[:, 1],'x', label='Acc1')
plt.plot(x, metrics_Variant_1[:, 2],'d', label='Acc2')
plt.plot(x, metrics_Variant_1[:, 3],'v', label='Acc3')
plt.ylim([0.0, 1.3])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel('Accuracy')
plt.title('Feautures list: Variant-4.1')

ax = plt.gca()
plt.tight_layout()

save_plot_file(plt, pathOUT, filename)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_meanDelt_class_Variant-4.1_regr"

plt.plot(x, metrics_Variant_1[:, 4],'o', label=r'$<\Delta>$ classes')

plt.ylim([0.0, 3.0])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta>$ classes')
plt.title('Feautures list: Variant-4.1')

ax = plt.gca()
plt.tight_layout()

save_plot_file(plt, pathOUT, filename)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_meanDelt_phiCPmix_Variant-4.1_regr"

plt.plot(x, metrics_Variant_1[:, 7],'o', label=r'$<\Delta \phi^{CP}>$ ')

plt.ylim([0.0, 1.0])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'$<\Delta \phi^{CP}>$ (rad)')
plt.title('Feautures list: Variant-4.1')

ax = plt.gca()
plt.tight_layout()

save_plot_file(plt, pathOUT, filename)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_L1delt_w_Variant-4.1_regr"

plt.plot(x, metrics_Variant_1[:, 5],'o', label=r'L1 $<\Delta w>$')

plt.ylim([0.0, 0.2])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'L1 $<\Delta w>$')
plt.title('Feautures list: Variant-4.1')

ax = plt.gca()
plt.tight_layout()

save_plot_file(plt, pathOUT, filename)

#---------------------------------------------------------------------

pathOUT = "figures/"
filename = "rhorho_L2delt_w_Variant-4.1_regr"

plt.plot(x, metrics_Variant_1[:, 6],'o', label=r'L2 $<\Delta w>$')

plt.ylim([0.0, 0.2])
plt.xticks(x)
plt.legend()
plt.xlabel('Number of classes')
plt.ylabel(r'L2 $<\Delta w>$')
plt.title('Feautures list: Variant-4.1')

ax = plt.gca()
plt.tight_layout()

save_plot_file(plt, pathOUT, filename)

#---------------------------------------------------------------------
 