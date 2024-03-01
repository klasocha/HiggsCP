import matplotlib.pyplot as plt
import os, errno
import numpy as np

def is_nan(x):
    """ Check whether x is NaN or not """
    return (x is np.nan or x != x)

def plot_two_TH1D(data, directory, filename, w_a, w_b , filt, step=0.05):
    """ TODO: Add description """

    data = data[filt]
    bins = int(1/step) + 1
    plt.hist([data, data], bins, weights=[w_a, w_b], 
             label=['scalar', 'pseudoscalar'], ls='dashed')
    
    plt.legend()
    plt.gca()
    plt.tight_layout()
    
    if filename:
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # PostScript .eps can also be used here
        plt.savefig(directory + filename + ".png")
	    # plt.savefig(directory + filename + ".eps")
    else:
        plt.show()
    
    plt.clf()



def plot_one_TH1D(data, directory, filename, w, filt, step=0.01):
    """ TODO: Add description """

    data = data[filt]
    bins = int(1/step) + 1
    plt.hist(data, bins, weights = w, ls='dashed')
    ax = plt.gca()
    ax.annotate("Mean = {:0.3f} \nRMS = {:1.3f}".format(np.mean(data), np.std(data)), 
                xy=(0.7, 0.85), 
                xycoords='axes fraction', 
                fontsize=12)
    plt.tight_layout()

    if filename:
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # PostScript .eps can also be used here
        plt.savefig(directory + filename+".png")
    else:
        plt.show()

    plt.clf()
 

