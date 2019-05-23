import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

labels = np.load('results/softmax_calc_w.npy')
preds = np.load('results/softmax_preds_w.npy')

i = 1
plt.plot(labels[i]/np.sum(labels[i]),'o')
plt.plot(preds[i],'o')
plt.show()
plt.clf()
