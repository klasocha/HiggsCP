import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

popts = np.load('../HiggsCP_data/rhorho/popts.npy')
pcovs = np.load('../HiggsCP_data/rhorho/pcovs.npy')

weights = np.load('../HiggsCP_data/rhorho/rhorho_raw.w.npy')

def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

x_weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * np.pi
x_fit = np.linspace(0, 2*np.pi)

i = 0 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 10 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 1000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 2000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 8000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

i = 10000 
plt.plot(x_weights, weights[:,i], 'o')
plt.plot(x_fit, weight_fun(x_fit, *popts[i]))
plt.xlabel('2 phi')
plt.ylabel('w')
print('error: ' + str(np.sqrt(np.diag(pcovs[i]))))

plt.show()
plt.clf()

