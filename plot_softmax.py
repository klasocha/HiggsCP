import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

calc_w  = np.load('monit_npy/nn_rhorho_Variant-All_Unweighted_False_FILTER_NUM_CLASSES_25/softmax_calc_w.npy')
preds_w = np.load('monit_npy/nn_rhorho_Variant-All_Unweighted_False_FILTER_NUM_CLASSES_25/softmax_preds_w.npy')

i = 1
plt.plot(calc_w[i]/np.sum(calc_w[i]),'o')
plt.plot(preds_w[i],'o')
plt.show()
plt.clf()

print  np.argmax(calc_w[:], axis=1)
print  np.argmax(calc_w[:], axis=0)
delta_argmax = np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)
plt.hist(delta_argmax, histtype='step', bins=100)
plt.show()
plt.clf()

acc0 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
