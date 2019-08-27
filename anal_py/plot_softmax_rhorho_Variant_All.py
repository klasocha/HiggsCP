import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_10/monit_npy/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
                           
i = 1
filename = "calc_preds_w_rhorho_Variant-All_nc_10_event_1"
x = np.arange(1,11)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.2])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
plt.title('Features list: Variant-All')
    
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
                           
i = 10
filename = "calc_preds_w_rhorho_Variant-All_nc_10_event_10"
x = np.arange(1,11)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.2])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
plt.title('Features list: Variant-All')
    
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
                           
i = 1000
filename = "calc_preds_w_rhorho_Variant-All_nc_10_event_1000"
x = np.arange(1,11)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.2])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
plt.title('Features list: Variant-All')
    
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
                           
i = 2000
filename = "calc_preds_w_rhorho_Variant-All_nc_10_event_2000"
x = np.arange(1,11)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.2])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
plt.title('Features list: Variant-All')
    
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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_4/monit_npy/"
pathOUT = "figures/"

calc_w_nc4  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc4 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc4 =  calculate_deltas_signed(np.argmax(preds_w_nc4[:], axis=1), np.argmax(calc_w_nc4[:], axis=1), 4)      

filename = "delt_argmax_rhorho_Variant-All_nc_4"
plt.hist(delt_argmax_nc4, histtype='step', bins=100)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc4=4
mean = np.mean(delt_argmax_nc4) * k2PI/nc4
std  = np.std(delt_argmax_nc4) * k2PI/nc4
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_10/monit_npy/"
pathOUT = "figures/"

calc_w_nc10  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc10 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc10 =  calculate_deltas_signed(np.argmax(preds_w_nc10[:], axis=1), np.argmax(calc_w_nc10[:], axis=1), 10)      

filename = "delt_argmax_rhorho_Variant-All_nc_10"
plt.hist(delt_argmax_nc10, histtype='step', bins=10)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc10=10
mean = np.mean(delt_argmax_nc10, dtype=np.float64)
std  = np.std(delt_argmax_nc10, dtype=np.float64)
meanrad = np.mean(delt_argmax_nc10, dtype=np.float64) * 6.28/10.0
stdrad  = np.std(delt_argmax_nc10, dtype=np.float64) * 6.28/10.0
ax.annotate("mean = {:0.3f} [idx] \nstd =  {:1.3f} [idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_20/monit_npy/"
pathOUT = "figures/"

calc_w_nc20  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc20 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc20 =  calculate_deltas_signed(np.argmax(preds_w_nc20[:], axis=1), np.argmax(calc_w_nc20[:], axis=1), 20)      

filename = "delt_argmax_rhorho_Variant-All_nc_20"
plt.hist(delt_argmax_nc20, histtype='step', bins=100)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc20=20
mean = np.mean(delt_argmax_nc20) * 6.28/20.0
std  = np.std(delt_argmax_nc20) * 6.28/20.0
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_25/monit_npy/"
pathOUT = "figures/"

calc_w_nc25  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc25 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc25 =  calculate_deltas_signed(np.argmax(preds_w_nc25[:], axis=1), np.argmax(calc_w_nc25[:], axis=1), 25)      

filename = "delt_argmax_rhorho_Variant-All_nc_25"
plt.hist(delt_argmax_nc25, histtype='step', bins=50)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc25=25
mean = np.mean(delt_argmax_nc25) * 6.28/25.0
std  = np.std(delt_argmax_nc25) * 6.28/25.0
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_50/monit_npy/"
pathOUT = "figures/"

calc_w_nc50  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc50 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc50 =  calculate_deltas_signed(np.argmax(preds_w_nc50[:], axis=1), np.argmax(calc_w_nc50[:], axis=1), 50)      

filename = "delt_argmax_rhorho_Variant-All_nc_50"
plt.hist(delt_argmax_nc50, histtype='step', bins=50)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
nc50=50
mean = np.mean(delt_argmax_nc50, dtype=np.float64)
std  = np.std(delt_argmax_nc50, dtype=np.float64)
meanrad = np.mean(delt_argmax_nc50, dtype=np.float64) * 6.28/50.0
stdrad  = np.std(delt_argmax_nc50, dtype=np.float64) * 6.28/50.0
ax.annotate("mean = {:0.3f} [idx] \nstd =  {:1.3f} [idx]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)
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
#----------------------------------------------------------------------------------

pathIN  = "../laptop_results/nn_rhorho_Variant-All_soft_Unweighted_False_NO_NUM_CLASSES_100/monit_npy/"
pathOUT = "figures/"

calc_w_nc100  = np.load(pathIN+'softmax_calc_w.npy')
preds_w_nc100 = np.load(pathIN+'softmax_preds_w.npy')
delt_argmax_nc100 = np.argmax(calc_w_nc100[:], axis=1) - np.argmax(preds_w_nc100[:], axis=1)
for i in range (len(delt_argmax_nc100)):
    if  delt_argmax_nc100[i] > 100.0/2.0 :
        delt_argmax_nc100[i] = 100.0 -  delt_argmax_nc100[i]
    if  delt_argmax_nc100[i] < - 100.0/2.0 :
        delt_argmax_nc100[i] = - 100.0 -  delt_argmax_nc100[i]

filename = "delt_argmax_rhorho_Variant-All_nc_100"
plt.hist(delt_argmax_nc100, histtype='step', bins=100)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc100) * 6.28/100.0
std  = np.std(delt_argmax_nc100) * 6.28/100.0
ax.annotate("mean = {:0.3f} [rad] \nstd =  {:1.3f} [rad]".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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
