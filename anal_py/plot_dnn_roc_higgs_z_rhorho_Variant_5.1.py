import sys
import os, errno
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

pathIN  =  "../laptop_results/nn_higgs_z_rhorho_Variant-5.1/"
pathOUT =  "figures/"

roc_curve_data = np.load(pathIN+'roc_curve_data.npy')
fpr = roc_curve_data[0]
tpr = roc_curve_data[1]
auc_score =  auc(fpr, tpr)


#----------------------------------------------------------------------------------

filename = "dnn_roc_curve_higgs_z_rhorho_Variant-5.1"

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Variant-5.1 (ROC AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
    
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
