import os, errno
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
pathOUT = "figures/"

calc_w  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w = np.load(pathIN+'test_regr_preds_weights.npy')

k2PI = 2*np.pi
#----------------------------------------------------------------------------------
                           
i = 1
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_51_event_1"
x = np.arange(1,52)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
#plt.title('Features list: Variant-All')
    
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
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_51_event_10"
x = np.arange(1,52)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.125])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
#plt.title('Features list: Variant-All')
    
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
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_51_event_1000"
x = np.arange(1,52)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
#plt.title('Features list: Variant-All')
    
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
filename = "regr_wt_calc_preds_rhorho_Variant-All_nc_51_event_2000"
x = np.arange(1,52)
plt.plot(x,calc_w[i], 'o', label='generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
#plt.ylim([0.0, 0.1])
plt.xlabel('Class index [idx]')
plt.xticks(x)
plt.ylabel(r'$wt$')
#plt.title('Features list: Variant-All')
    
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

#----------------------------------------------------------------------------------

pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_51/monit_npy/"
pathOUT = "figures/"

calc_w_nc51  = np.load(pathIN+'test_regr_calc_weights.npy')
preds_w_nc51 = np.load(pathIN+'test_regr_preds_weights.npy')
delt_argmax_nc51 =  calculate_deltas_signed(np.argmax(preds_w_nc51[:], axis=1), np.argmax(calc_w_nc51[:], axis=1), 51)      

filename = "regr_wt_delt_argmax_rhorho_Variant-All_nc_51"
plt.hist(delt_argmax_nc51, histtype='step', bins=51)
plt.xlabel(r'$\Delta_{class} [idx]$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
nc51=51
mean = np.mean(delt_argmax_nc51, dtype=np.float64)
std  = np.std(delt_argmax_nc51, dtype=np.float64)
meanerr = stats.sem(delt_argmax_nc51)
meanrad = np.mean(delt_argmax_nc51, dtype=np.float64) * k2PI/nc51
stdrad  = np.std(delt_argmax_nc51, dtype=np.float64) * k2PI/nc51
meanerrrad = stats.sem(delt_argmax_nc51) * k2PI/nc51
ax.annotate("mean = {:0.3f}+- {:1.3f}[idx] \nstd =  {:1.3f} [idx]".format(mean,meanerr, std ), xy=(0.56, 0.85), xycoords='axes fraction', fontsize=12)
ax.annotate("mean = {:0.3f}+- {:1.3f}[rad] \nstd =  {:1.3f} [rad]".format(meanrad,meanerrrad, stdrad ), xy=(0.56, 0.70), xycoords='axes fraction', fontsize=12)

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
