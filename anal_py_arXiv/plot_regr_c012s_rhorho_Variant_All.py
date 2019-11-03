import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize

from anal_utils import weight_fun, calc_weights
from src_py.metrics_utils import  calculate_deltas_signed



pathIN  = "../laptop_results_dropout=0/nn_rhorho_Variant-All_regr_c012s_hits_c0s_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_c012s  = np.load(pathIN+'test_regr_calc_c012s.npy')
preds_c012s = np.load(pathIN+'test_regr_preds_c012s.npy')

        
k2PI = 2 * np.pi
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 1
filename = "regr_c012s_calc_preds_rhorho_Variant-All_event_1"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_c012s[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_c012s[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
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
# ERW
# should normalise to same area which is not the case for now
                           
i = 10
filename = "regr_c012s_calc_preds_rhorho_Variant-All_event_10"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_c012s[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_c012s[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
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
# ERW
# should normalise to same area which is not the case for now
                           
i = 100
filename = "regr_c012s_calc_preds_rhorho_Variant-All_event_100"
x = np.linspace(0, k2PI, 100)
plt.plot(x,weight_fun(x, *calc_c012s[i]), 'o', label='generated')
plt.plot(x,weight_fun(x, *preds_c012s[i]), 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 3.0])
plt.xlabel(r'$\alpha^{CP}$ [rad]')
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

delt_c012s= calc_c012s - preds_c012s

print calc_c012s[:,0]
print preds_c012s[:,0]
print delt_c012s[:,0]
#----------------------------------------------------------------------------------

filename = "regr_c012s_delt_C0_rhorho_Variant-All"
plt.hist(delt_c012s[:,0], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta C_{0}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c012s[:,0])
std  = np.std(delt_c012s[:,0])
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

filename = "regr_c012s_C0_rhorho_Variant-All"
plt.hist(calc_c012s[:,0], histtype='step', color = 'black', linestyle='--', bins=50)
plt.hist(preds_c012s[:,0], histtype='step', color = 'red', bins=50)
plt.xlim([-0.0, 2.0])
plt.xlabel(r'$C_{0}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_c012s[:,0], dtype=np.float64)
calc_std  = np.std(calc_c012s[:,0], dtype=np.float64)
preds_mean = np.mean(preds_c012s[:,0], dtype=np.float64)
preds_std  = np.std(preds_c012s[:,0], dtype=np.float64)
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
print calc_c012s[:,1]
print preds_c012s[:,1]
print delt_c012s[:,1]
#----------------------------------------------------------------------------------

filename = "regr_c012s_C1_rhorho_Variant-All"
plt.hist(calc_c012s[:,1], histtype='step', bins=50, linestyle='--', color = 'black')
plt.hist(preds_c012s[:,1], histtype='step', bins=50, color = 'red')
plt.xlim([-2.0, 2.0])
plt.xlabel(r'$C_{1}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_c012s[:,1],dtype=np.float64)
calc_std  = np.std(calc_c012s[:,1],dtype=np.float64)
preds_mean = np.mean(preds_c012s[:,1],dtype=np.float64)
preds_std  = np.std(preds_c012s[:,1],dtype=np.float64)
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

filename = "regr_c012s_delt_C1_rhorho_Variant-All"
plt.hist(delt_c012s[:,1], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta C_{1}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c012s[:,1],dtype=np.float64)
std  = np.std(delt_c012s[:,1],dtype=np.float64)
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
print calc_c012s[:,2]
print preds_c012s[:,2]
print delt_c012s[:,2]
#----------------------------------------------------------------------------------

filename = "regr_c012s_C2_rhorho_Variant-All"
plt.hist(calc_c012s[:,2], histtype='step', color = 'black', linestyle='--', bins=50)
plt.hist(preds_c012s[:,2], histtype='step',  color = 'red', bins=50)
plt.xlim([-2.0, 2.0])
plt.xlabel(r'$C_{2}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
calc_mean = np.mean(calc_c012s[:,2],dtype=np.float64)
calc_std  = np.std(calc_c012s[:,2],dtype=np.float64)
preds_mean = np.mean(preds_c012s[:,2],dtype=np.float64)
preds_std  = np.std(preds_c012s[:,2],dtype=np.float64)
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

filename = "regr_c012s_delt_C2_rhorho_Variant-All"
plt.hist(delt_c012s[:,2], histtype='step', bins=50,  color = 'black')
plt.xlim([-1.5, 1.5])
plt.xlabel(r'$\Delta C_{2}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_c012s[:,2],dtype=np.float64)
std  = np.std(delt_c012s[:,2],dtype=np.float64)
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

calc_w_nc5  =  calc_weights(5, calc_c012s)
preds_w_nc5 =  calc_weights(5, preds_c012s)
delt_argmax_nc5 =  calculate_deltas_signed(np.argmax(preds_w_nc5[:], axis=1), np.argmax(calc_w_nc5[:], axis=1), 5)      
nc5=5.0

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_5"
plt.hist(delt_argmax_nc5, histtype='step', bins=5)
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc5) * k2PI/nc5
std  = np.std(delt_argmax_nc5) * k2PI/nc5
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
calc_w_nc11  =  calc_weights(11, calc_c012s)
preds_w_nc11 =  calc_weights(11, preds_c012s)
delt_argmax_nc11 =  calculate_deltas_signed(np.argmax(preds_w_nc11[:], axis=1), np.argmax(calc_w_nc11[:], axis=1), 11)      

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_11"
plt.hist(delt_argmax_nc11, histtype='step', bins=100)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc11)
std  = np.std(delt_argmax_nc11)
meanrad = np.mean(delt_argmax_nc11) * k2PI/11.0
stdrad  = np.std(delt_argmax_nc11) * k2PI/11.0
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
calc_w_nc11  =  calc_weights(11, calc_c012s)
preds_w_nc11 =  calc_weights(11, preds_c012s)
preds_argmax_nc11 =  np.argmax(preds_w_nc11[:], axis=1) * k2PI/11.0    
calc_argmax_nc11  =  np.argmax(calc_w_nc11[:], axis=1) * k2PI/11.0    

filename = "regr_c012s_calc_argmax_rhorho_Variant-All_nc_11"
plt.hist(calc_argmax_nc11, histtype='step', color = "black", bins=100, label = "generated")
plt.hist(preds_argmax_nc11, histtype='step', color = "red", bins=100, label = "predicted")
plt.ylabel('Entries')
plt.xlabel(r'$\alpha^{CP}_{max}$')
#plt.title('Features list: Variant-All')

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
calc_w_nc21  =  calc_weights(21, calc_c012s)
preds_w_nc21 =  calc_weights(21, preds_c012s)
delt_argmax_nc21 =  calculate_deltas_signed(np.argmax(preds_w_nc21[:], axis=1), np.argmax(calc_w_nc21[:], axis=1), 21)      

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_21"
plt.hist(delt_argmax_nc21, histtype='step', bins=21)
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc21) * k2PI/21.0
std  = np.std(delt_argmax_nc21) * k2PI/21.0
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
calc_w_nc25  =  calc_weights(25, calc_c012s)
preds_w_nc25 =  calc_weights(25, preds_c012s)
delt_argmax_nc25 =  calculate_deltas_signed(np.argmax(preds_w_nc25[:], axis=1), np.argmax(calc_w_nc25[:], axis=1), 25)      

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_25"
plt.hist(delt_argmax_nc25, histtype='step', bins=25)
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc25) * k2PI/25.0
std  = np.std(delt_argmax_nc25) * k2PI/25.0
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
calc_w_nc51  =  calc_weights(51, calc_c012s)
preds_w_nc51 =  calc_weights(51, preds_c012s)
delt_argmax_nc51 =  calculate_deltas_signed(np.argmax(preds_w_nc51[:], axis=1), np.argmax(calc_w_nc51[:], axis=1), 51)      

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_51"
plt.hist(delt_argmax_nc51, histtype='step', bins=51)
plt.ylabel('Entries')
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc51)
std  = np.std(delt_argmax_nc51)
meanrad = np.mean(delt_argmax_nc51) * k2PI/51.0
stdrad  = np.std(delt_argmax_nc51) * k2PI/51.0
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
calc_w_nc101  =  calc_weights(101, calc_c012s)
preds_w_nc101 =  calc_weights(101, preds_c012s)
preds_argmax_nc101 =  np.argmax(preds_w_nc101[:], axis=1) * k2PI/101.0    
calc_argmax_nc101 =  np.argmax(calc_w_nc101[:], axis=1) * k2PI/101.0    

filename = "regr_c012s_calc_argmax_rhorho_Variant-All_nc_101"
plt.hist(preds_argmax_nc101, histtype='step', bins=101)
plt.ylabel('Entries')
plt.xlabel(r'$\alpha^{CP}_{max}$')
#plt.title('Features list: Variant-All')

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

filename = "regr_c012s_calc_preds_argmax_rhorho_Variant-All_nc_101"
plt.hist(calc_argmax_nc101, histtype='step', color = "black", bins=50, label = "generated")
plt.hist(preds_argmax_nc101, histtype='step', color = "red", bins=50, label = "predicted")
#plt.ylim([0, 800])
plt.ylabel('Entries')
plt.xlabel(r'$\alpha^{CP}_{max}$')
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
calc_w_nc101  =  calc_weights(101, calc_c012s)
preds_w_nc101 =  calc_weights(101, preds_c012s)
delt_argmax_nc101 =  calculate_deltas_signed(np.argmax(preds_w_nc101[:], axis=1), np.argmax(calc_w_nc101[:], axis=1), 101)      

filename = "regr_c012s_delt_argmax_rhorho_Variant-All_nc_101"
plt.hist(delt_argmax_nc101, histtype='step', bins=101)
plt.xlabel(r'$\Delta_{class}$')
#plt.title('Features list: Variant-All')

ax = plt.gca()
mean = np.mean(delt_argmax_nc101) * k2PI/101.0
std  = np.std(delt_argmax_nc101) * k2PI/101.0
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
