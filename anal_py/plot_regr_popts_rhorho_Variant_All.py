import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize

from anal_utils import weight_fun, calc_weights
from src_py.metrics_utils import  calculate_deltas_signed



pathIN  = "../laptop_results/nn_rhorho_Variant-All_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_popts  = np.load(pathIN+'valid_regr_calc_popts.npy')
preds_popts = np.load(pathIN+'valid_regr_preds_popts.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 1
filename = "regr_preds_popts_rhorho_Variant-All_event_1"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel(r'$wt$')
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
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 10
filename = "regr_preds_popts_rhorho_Variant-All_event_10"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel(r'$w$')
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
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 100
filename = "regr_preds_popts_rhorho_Variant-All_event_100"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
plt.ylabel('w')
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

delt_popts= calc_popts - preds_popts

print calc_popts[:,0]
print delt_popts[:,0]
print delt_popts[:,0]/calc_popts[:,0]

filename = "delt_popts_A_rhorho_Variant-All"
plt.hist(delt_popts[:,0], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$A')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_popts[:,0])
std  = np.std(delt_popts[:,0])
ax.annotate("mean = {:0.3f} \nstd =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

delt_popts= calc_popts - preds_popts

print calc_popts[:,0]
print delt_popts[:,0]
print delt_popts[:,0]/calc_popts[:,0]
#----------------------------------------------------------------------------------

filename = "popts_A_rhorho_Variant-All"
plt.hist(calc_popts[:,0], histtype='step', color = 'black', linestyle='--', bins=50)
plt.hist(preds_popts[:,0], histtype='step', color = 'red', bins=50)
plt.xlim([-0.0, 2.0])
plt.xlabel(r'A')
plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_popts[:,0], dtype=np.float64)
calc_std  = np.std(calc_popts[:,0], dtype=np.float64)
preds_mean = np.mean(preds_popts[:,0], dtype=np.float64)
preds_std  = np.std(preds_popts[:,0], dtype=np.float64)
ax.annotate("Gener.:  mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.60, 0.85), xycoords='axes fraction', fontsize=12, color = 'black')
ax.annotate("Pred. :  mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.60, 0.65), xycoords='axes fraction', fontsize=12, color = 'red')


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
print calc_popts[:,1]
print delt_popts[:,1]
print delt_popts[:,1]/calc_popts[:,1]
#----------------------------------------------------------------------------------

filename = "popts_B_rhorho_Variant-All"
plt.hist(calc_popts[:,1], histtype='step', bins=50, linestyle='--', color = 'black')
plt.hist(preds_popts[:,1], histtype='step', bins=50, color = 'red')
plt.xlim([-2.0, 2.0])
plt.xlabel(r'B')
plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_popts[:,1],dtype=np.float64)
calc_std  = np.std(calc_popts[:,1],dtype=np.float64)
preds_mean = np.mean(preds_popts[:,1],dtype=np.float64)
preds_std  = np.std(preds_popts[:,1],dtype=np.float64)
ax.annotate("Gener.:  mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.60, 0.85), xycoords='axes fraction', fontsize=12, color = 'black')
ax.annotate("Pred.:   mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.60, 0.65), xycoords='axes fraction', fontsize=12, color = 'red')

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

filename = "delt_popts_B_rhorho_Variant-All"
plt.hist(delt_popts[:,1], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$B')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_popts[:,1],dtype=np.float64)
std  = np.std(delt_popts[:,1],dtype=np.float64)
ax.annotate("mean = {:0.3f} \nstd =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)


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
print calc_popts[:,2]
print delt_popts[:,2]
print delt_popts[:,2]/calc_popts[:,2]
#----------------------------------------------------------------------------------

filename = "popts_C_rhorho_Variant-All"
plt.hist(calc_popts[:,2], histtype='step', color = 'black', linestyle='--', bins=50)
plt.hist(preds_popts[:,2], histtype='step',  color = 'red', bins=50)
plt.xlim([-2.0, 2.0])
plt.xlabel(r'C')
plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_popts[:,2],dtype=np.float64)
calc_std  = np.std(calc_popts[:,2],dtype=np.float64)
preds_mean = np.mean(preds_popts[:,2],dtype=np.float64)
preds_std  = np.std(preds_popts[:,2],dtype=np.float64)
ax.annotate("Gener.: mean = {:0.3f}, \n           std =  {:1.3f}".format(calc_mean, calc_std), xy=(0.60, 0.85), xycoords='axes fraction', fontsize=12, color = 'black')
ax.annotate("Pred. : mean = {:0.3f}, \n           std =  {:1.3f}".format(preds_mean, preds_std), xy=(0.60, 0.65), xycoords='axes fraction', fontsize=12, color = 'red')

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

filename = "delt_popts_C_rhorho_Variant-All"
plt.hist(delt_popts[:,2], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta$C')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_popts[:,2],dtype=np.float64)
std  = np.std(delt_popts[:,2],dtype=np.float64)
ax.annotate("mean = {:0.3f} \nstd =  {:1.3f}".format(mean, std), xy=(0.65, 0.85), xycoords='axes fraction', fontsize=12)

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

calc_w_nc4  =  calc_weights(4, calc_popts)
preds_w_nc4 =  calc_weights(4, preds_popts)
delt_argmax_nc4 =  calculate_deltas_signed(np.argmax(preds_w_nc4[:], axis=1), np.argmax(calc_w_nc4[:], axis=1), 4)      
nc4=4.0

filename = "delt_argmax_rhorho_Variant-All_nc_4_regr"
plt.hist(delt_argmax_nc4, histtype='step', bins=100)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
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
#----------------------------------------------------------------------------------
calc_w_nc10  =  calc_weights(10, calc_popts)
preds_w_nc10 =  calc_weights(10, preds_popts)
delt_argmax_nc10 =  calculate_deltas_signed(np.argmax(preds_w_nc10[:], axis=1), np.argmax(calc_w_nc10[:], axis=1), 10)      

filename = "delt_argmax_rhorho_Variant-All_nc_10_regr"
plt.hist(delt_argmax_nc10, histtype='step', bins=10)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc10)
std  = np.std(delt_argmax_nc10)
meanrad = np.mean(delt_argmax_nc10) * 6.28/10.0
stdrad  = np.std(delt_argmax_nc10) * 6.28/10.0
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
calc_w_nc20  =  calc_weights(20, calc_popts)
preds_w_nc20 =  calc_weights(20, preds_popts)
delt_argmax_nc20 =  calculate_deltas_signed(np.argmax(preds_w_nc20[:], axis=1), np.argmax(calc_w_nc20[:], axis=1), 20)      

filename = "delt_argmax_rhorho_Variant-All_nc_20_regr"
plt.hist(delt_argmax_nc20, histtype='step', bins=100)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
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
calc_w_nc25  =  calc_weights(25, calc_popts)
preds_w_nc25 =  calc_weights(25, preds_popts)
delt_argmax_nc25 =  calculate_deltas_signed(np.argmax(preds_w_nc25[:], axis=1), np.argmax(calc_w_nc25[:], axis=1), 25)      

filename = "delt_argmax_rhorho_Variant-All_nc_25_regr"
plt.hist(delt_argmax_nc25, histtype='step', bins=50)
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
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
calc_w_nc50  =  calc_weights(50, calc_popts)
preds_w_nc50 =  calc_weights(50, preds_popts)
delt_argmax_nc50 =  calculate_deltas_signed(np.argmax(preds_w_nc50[:], axis=1), np.argmax(calc_w_nc50[:], axis=1), 50)      

filename = "delt_argmax_rhorho_Variant-All_nc_50_regr"
plt.hist(delt_argmax_nc50, histtype='step', bins=50)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc50)
std  = np.std(delt_argmax_nc50)
meanrad = np.mean(delt_argmax_nc50) * 6.28/50.0
stdrad  = np.std(delt_argmax_nc50) * 6.28/50.0
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
calc_w_nc100  =  calc_weights(100, calc_popts)
preds_w_nc100 =  calc_weights(100, preds_popts)
delt_argmax_nc100 =  calculate_deltas_signed(np.argmax(preds_w_nc100[:], axis=1), np.argmax(calc_w_nc100[:], axis=1), 100)      

filename = "delt_argmax_rhorho_Variant-All_nc_100_regr"
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
