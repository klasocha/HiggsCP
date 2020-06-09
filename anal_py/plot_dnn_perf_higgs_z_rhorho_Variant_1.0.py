import sys
import os, errno
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from scipy import optimize


pathIN  = "../laptop_results/nn_higgs_z_rhorho_Variant-1.0/"
pathOUT = "figures/"

train_losses    = np.load(pathIN+'train_losses.npy')
valid_losses    = np.load(pathIN+'valid_losses.npy')
test_losses     = np.load(pathIN+'test_losses.npy')

train_aucs      = np.load(pathIN+'train_aucs.npy')
test_aucs       = np.load(pathIN+'test_aucs.npy')
valid_aucs      = np.load(pathIN+'valid_aucs.npy')

#----------------------------------------------------------------------------------

filename = "dnn_train_loss_higgs_z_rhorho_Variant-1.0"
x = np.arange(1,len(train_losses)+1)
plt.plot([0], marker='None',
           linestyle='None', label=r'Classification: Higgs vs Z')
plt.plot(x,train_losses, 'o', color = 'black', label='Training')
plt.plot(x,valid_losses, 'd', color = 'orange', label='Validation')
plt.legend()
plt.ylim([0.800,1.000])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel('Loss')
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

filename = "dnn_train_aucs_higgs_z_rhorho_Variant-1.0"
x = np.arange(1,len(train_aucs)+1)
plt.plot(x,train_aucs, 'o', label='training')
plt.plot(x,valid_aucs, 'd', label='validation')
plt.legend()
#plt.ylim([0.0, 0.4])
plt.xlabel('Number of epochs')
plt.xticks(x)
plt.ylabel(r'aucs')
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
