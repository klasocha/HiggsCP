import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
calc_hits_c0s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c0s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c1s_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
calc_hits_c1s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c1s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_soft_c012s_hits_c2s_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
calc_hits_c2s   = np.load(pathIN+'train_soft_calc_hits_c012s.npy')
preds_hits_c2s  = np.load(pathIN+'train_soft_preds_hits_c012s.npy')

pathOUT = "figures/"


data_len = calc_hits_c0s.shape[0]
preds_c0s = np.zeros((data_len, 1))
calc_c0s = np.zeros((data_len, 1))
preds_c1s = np.zeros((data_len, 1))
calc_c1s = np.zeros((data_len, 1))
preds_c2s = np.zeros((data_len, 1))
calc_c2s = np.zeros((data_len, 1))

for i in range(data_len):
    preds_c0s[i] = np.argmax(preds_hits_c0s[i])
    calc_c0s[i] = np.argmax(calc_hits_c0s[i])
    preds_c1s[i] = np.argmax(preds_hits_c1s[i])
    calc_c1s[i] = np.argmax(calc_hits_c1s[i])
    preds_c2s[i] = np.argmax(preds_hits_c2s[i])
    calc_c2s[i] = np.argmax(calc_hits_c2s[i])
#    print i, " ",  preds_hits_c0s[i]
#    print i, " ",  preds_hits_c1s[i]
#    print preds_c0s[i], " ", preds_c1s[i]

    
print calc_hits_c0s[0]
print calc_c0s[0]
print preds_hits_c0s[0]
print preds_c0s[0]
    
print calc_hits_c1s[0]
print calc_c1s[0]
print preds_hits_c1s[0]
print preds_c1s[0]
    
print calc_hits_c2s[0]
print calc_c2s[0]
print preds_hits_c2s[0]
print preds_c2s[0]


delt_c0s =  calculate_deltas_signed(np.argmax(preds_hits_c0s[:], axis=1), np.argmax(calc_hits_c0s[:], axis=1), 51)      
delt_c1s =  calculate_deltas_signed(np.argmax(preds_hits_c1s[:], axis=1), np.argmax(calc_hits_c1s[:], axis=1), 51)      
delt_c2s =  calculate_deltas_signed(np.argmax(preds_hits_c2s[:], axis=1), np.argmax(calc_hits_c2s[:], axis=1), 51)      


k2PI= 2* np.pi
#----------------------------------------------------------------------------------
filename = "soft_c012s_c0s_rhorho_Variant-All_nc51"

plt.hist(calc_c0s, histtype='step', bins=50,  color = 'black', label = "generated")
plt.hist(preds_c0s, histtype='step', bins=50,  color = 'red', label = "predicted")
plt.xlim([0, 50.0])
plt.ylim([0, 8000])
plt.xlabel(r'Class index')
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
filename = "soft_c012s_delt_c0s_rhorho_Variant-All_nc51"

plt.hist(delt_c0s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c0s)
std  = np.std(delt_c0s)
meanrad = np.mean(delt_c0s) * k2PI/51.0
stdrad  = np.std(delt_c0s) * k2PI/51.0
ax.annotate("mean = {:0.3f}[idx] \nstd =  {:1.3f}[idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(meanrad, stdrad), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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
filename = "soft_c012s_c1s_rhorho_Variant-All_nc51"

plt.hist(calc_c1s, histtype='step', bins=50,  color = 'black', label = "generated")
plt.hist(preds_c1s, histtype='step', bins=50,  color = 'red', label = "predicted")
plt.xlim([0, 50.0])
plt.ylim([0, 5000])
plt.xlabel(r'Class index')
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
filename = "soft_c012s_delt_c1s_rhorho_Variant-All_nc51"

plt.hist(delt_c1s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c1s)
std  = np.std(delt_c1s)
meanrad = np.mean(delt_c1s) * k2PI/51.0
stdrad  = np.std(delt_c1s) * k2PI/51.0
ax.annotate("mean = {:0.3f}[idx] \nstd =  {:1.3f}[idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(meanrad, stdrad), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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
filename = "soft_c012s_c2s_rhorho_Variant-All_nc51"

plt.hist(calc_c2s, histtype='step', bins=50,  color = 'black', label = "generated")
plt.hist(preds_c2s, histtype='step', bins=50,  color = 'red', label = "predicted")
plt.xlim([0, 50.0])
plt.ylim([0, 5000])
plt.xlabel(r'Class index')
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
filename = "soft_c012s_delt_c2s_rhorho_Variant-All_nc51"

plt.hist(delt_c2s, histtype='step', bins=51,  color = 'black')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c2s)
std  = np.std(delt_c2s)
meanrad = np.mean(delt_c2s) * k2PI/51.0
stdrad  = np.std(delt_c2s) * k2PI/51.0
ax.annotate("mean = {:0.3f}[idx] \nstd =  {:1.3f}[idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(meanrad, stdrad), xy=(0.65, 0.65), xycoords='axes fraction', fontsize=12)

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