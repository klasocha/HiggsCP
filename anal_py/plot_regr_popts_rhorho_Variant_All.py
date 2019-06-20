import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize

from anal_utils import weight_fun



pathIN  = "npy/nn_rhorho_Variant-All_regr_popts_Unweighted_False_NO_NUM_CLASSES_0/monit_npy/"
pathOUT = "figures/"

calc_popts  = np.load(pathIN+'valid_regr_calc_popts.npy')
preds_popts = np.load(pathIN+'valid_regr_preds_popts.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 1
filename = "regr_preds_popts_rhorho_Variant-All_event_1"
x = np.linspace(0, 2*np.pi, 10)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel('CP mixing parameter')
plt.xticks(x)
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
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 10
filename = "regr_preds_popts_rhorho_Variant-All_event_10"
x = np.linspace(0, 2*np.pi, 10)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel('CP mixing parameter')
plt.xticks(x)
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
#----------------------------------------------------------------------------------
# ERW
# should normalise to same area which is not the case for now
                           
i = 100
filename = "regr_preds_popts_rhorho_Variant-All_event_100"
x = np.linspace(0, 2*np.pi, 10)
plt.plot(x,weight_fun(x, *calc_popts[i]), 'o', label='calc_w')
plt.plot(x,weight_fun(x, *preds_popts[i]), 'd', label='preds_w')
plt.legend()
plt.ylim([0.0, 2.0])
plt.xlabel('CP mixing parameter')
plt.xticks(x)
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
