import matplotlib.pyplot as plt
import os, errno
import numpy as np

def is_nan(x):
    return (x is np.nan or x != x)

DIRECTORY = "../monit_plots/"

def feature_plot(data, step=0.05, directory = None, filename=None, title=None, Xlabel=None, Ylabel=None, w_a = None, w_b = None, filt = None):

        
        if filt is not None:
            filt = [x==1 for x in filt]
            data = data[filt]
            w_a = w_a[filt]
            w_b = w_b[filt]
	else:
            filt = [not is_nan(x) for x in data]
            data = data[filt]
            w_a = w_a[filt]
            w_b = w_b[filt]

	bins = int(1/step) + 1

	plt.hist([data, data], bins, weights=[w_a, w_b], label=['scalar', 'pseudoscalar'], ls='dashed')
	
        plt.legend()
        ax = plt.gca()
	if title:
		ax.annotate(title, xy=(0.6, 0.9), xycoords='axes fraction', fontsize=12)
	ax.set_xlabel(Xlabel, fontsize=12)
	ax.set_ylabel(Ylabel, fontsize=12)
	plt.tight_layout()

	if filename:
                try:
                    os.makedirs(directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
		plt.savefig(directory + filename+".eps")
	else:
		plt.show()
	plt.clf()
