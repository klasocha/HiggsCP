import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

pathIN  = "../laptop_results/monit_npy/nn_rhorho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

#----------------------------------------------------------------------------------

#ERW
# why it is plotting two dots in the legend box?

i = 1
filename = "preds_a1rho_w_event_1"
plt.plot(calc_w[i], 'o', label='calc_w')
plt.plot(preds_w[i], 'o', label='preds_w')
plt.xlabel('phiCP class')
plt.ylabel('w')
plt.legend()
    
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
plt.clf()

#----------------------------------------------------------------------------------

print  np.argmax(calc_w[:], axis=1)
print  np.argmax(calc_w[:], axis=0)
delt_argmax = np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)

filename = "delt_argmax_a1rho"
plt.hist(delt_argmax, histtype='step', bins=100)
plt.xlabel('delt phiCP class')
plt.legend()

ax = plt.gca()
ax.annotate("Mean = {:0.3f} \nRMS = {:1.3f}".format(np.mean(delt_argmax), np.std(delt_argmax)), xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)

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

acc0 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
