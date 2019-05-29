import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize

pathIN  = "../laptop_results/monit_npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10/"
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
plt.title('Features list: Variant-All')
    
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
plt.xlabel(r'$\Delta$  classes')
plt.title('Features list: Variant-All')

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

plt.clf()

acc0 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
#----------------------------------------------------------------------------------

pathIN  = "../laptop_results/monit_npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_4/"
pathOUT = "figures/"

calc_w_nc4  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc4 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc4 = np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)

filename = "delt_argmax_a1rho_Variant-All_nc_4"
plt.hist(delt_argmax_nc4, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  classes')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc4) * 6.28/4.0
rms  = np.std(delt_argmax_nc4) * 6.28/4.0
ax.annotate("Mean = {:0.3f} (rad) \nRMS =  {:1.3f} (rad)".format(mean, rms), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

plt.clf()

acc0 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc4[:], axis=1) - np.argmax(preds_w_nc4[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
#----------------------------------------------------------------------------------

pathIN  = "../laptop_results/monit_npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_10/"
pathOUT = "figures/"

calc_w_nc10  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc10 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc10 = np.argmax(calc_w[:], axis=1) - np.argmax(preds_w[:], axis=1)

filename = "delt_argmax_a1rho_Variant-All_nc_10"
plt.hist(delt_argmax, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  classes')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc10) * 6.28/10.0
rms  = np.std(delt_argmax_nc10) * 6.28/10.0
ax.annotate("Mean = {:0.3f} (rad) \nRMS =  {:1.3f} (rad)".format(mean, rms), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

plt.clf()

acc0 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc10[:], axis=1) - np.argmax(preds_w_nc10[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
#----------------------------------------------------------------------------------

pathIN  = "../laptop_results/monit_npy/nn_a1rho_Variant-All_Unweighted_False_NO_NUM_CLASSES_20/"
pathOUT = "figures/"

calc_w_nc20  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc20 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc20 = np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)

filename = "delt_argmax_a1rho_Variant-All_nc_20"
plt.hist(delt_argmax_nc20, histtype='step', bins=100)
plt.xlabel(r'$\Delta$  classes')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc20) * 6.28/20.0
rms  = np.std(delt_argmax_nc20) * 6.28/20.0
ax.annotate("Mean = {:0.3f} (rad) \nRMS =  {:1.3f} (rad)".format(mean, rms), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

plt.clf()

acc0 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 0).mean()
acc1 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 1).mean()
acc2 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 2).mean()
acc3 = (np.abs(np.argmax(calc_w_nc20[:], axis=1) - np.argmax(preds_w_nc20[:], axis=1)) <= 3).mean()
print('Acc0', acc0)
print('Acc1', acc1)
print('Acc2', acc2)
print('Acc3', acc3)
#----------------------------------------------------------------------------------
