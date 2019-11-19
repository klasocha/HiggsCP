import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy import optimize

from anal_utils import weight_fun, calc_weights
from src_py.metrics_utils import calculate_deltas_signed_pi

pathIN = "../temp_results/nn_rhorho_Variant-All_regr_argmaxs_hits_c0s_Unweighted_False_FILTER_NUM_CLASSES_21/monit_npy/"
pathOUT = "figures/"

calc_argmaxs = np.load(pathIN + 'test_regr_calc_argmaxs.npy')
preds_argmaxs = np.load(pathIN + 'test_regr_preds_argmaxs.npy')

preds_argmaxs += ((preds_argmaxs < 0) * 2 * np.pi)
preds_argmaxs += ((preds_argmaxs < 0) * 2 * np.pi)

preds_argmaxs -= ((preds_argmaxs > (2 * np.pi)) * 2 * np.pi)

print calc_argmaxs
print preds_argmaxs
print calc_argmaxs - preds_argmaxs

delt_argmaxxs = calc_argmaxs - preds_argmaxs
delt_argmaxs = calculate_deltas_signed_pi(calc_argmaxs, preds_argmaxs)

print "MEAN ERROR", np.mean(np.minimum(np.abs(calc_argmaxs - preds_argmaxs), 2 * np.pi - np.abs(calc_argmaxs - preds_argmaxs)))

k2PI = 2 * np.pi
#calc_argmaxs= calc_argmaxs/k2PI
#print calc_argmaxs

#----------------------------------------------------------------------------------
filename = "regr_argmaxs_calc_preds_argmax_rhorho_Variant-All"

plt.hist(calc_argmaxs, histtype='step', bins=50,  color = 'black', linestyle='--', label="Generated")
plt.hist(preds_argmaxs, histtype='step', bins=50, color = 'red', label=r"Regression: $\alpha^{CP}_{max}$")
plt.xlim([0, k2PI])
plt.ylim([0, 1700])
plt.xlabel(r'$\alpha^{CP}_{max}$[rad]')
#plt.title('Features list: Variant-All')
plt.legend()

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

plt.clf()
#----------------------------------------------------------------------------------
filename = "regr_argmaxs_delt_argmax_rhorho_Variant-All"

plt.hist(delt_argmaxs, histtype='step', bins=50,  color = 'black')
plt.xlim([-3.2, 3.2])
plt.xlabel(r'$\Delta \alpha^{CP}_{max}$ [rad]')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)

table_vals=[[r"Regression: $\alpha^{CP}_{max}$"],
            [" "],
            ["mean = {:0.3f} [rad]".format(mean)],
            ["std = {:1.3f} [rad]".format(std)]
            ]

table = plt.table(cellText=table_vals,
                  colWidths = [0.30],
                  cellLoc="left",
                  loc='upper right')
table.set_fontsize(12)

for key, cell in table.get_celld().items():
    cell.set_linewidth(0)

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

plt.clf()
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
filename = "delt_regr_argmaxs_rhorho_Variant-All"

plt.hist(delt_argmaxs, histtype='step', bins=50,  color = 'black')
plt.xlim([-np.pi, np.pi])
plt.xlabel(r'$\Delta \alpha^{CP}_{max}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmaxs)
std  = np.std(delt_argmaxs)
ax.annotate("mean = {:0.3f}[rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

plt.clf()
#----------------------------------------------------------------------------------