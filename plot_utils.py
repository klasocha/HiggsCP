import matplotlib.pyplot as plt
import os, errno
import numpy as np

def is_nan(x):
    return (x is np.nan or x != x)

DIRECTORY = "../monit_plots/"

def feature_plot(data, directory, filename, w_a, w_b , filt,step=0.05):

    data = data[filt]

    bins = int(1/step) + 1
    
    plt.hist([data, data], bins, weights=[w_a, w_b], label=['scalar', 'pseudoscalar'], ls='dashed')
    
    plt.legend()
    ax = plt.gca()
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

def monit_plots(args, event, w_a, w_b):

    if args.PLOT_FEATURES == "FILTER":
        filt = [x==1 for x in event.cols[:,-1]]
        w_a = w_a[filt]
        w_b = w_b[filt]
    else:
        filt = [not is_nan(x) for x in data]
        w_a = w_a[filt]
        w_b = w_b[filt]

    for i in range(len(event.cols[0,:])-1):
        feature_plot(event.cols[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = event.labels[i], w_a = w_a, w_b = w_b, filt = filt)
    for i in range(len(event.labels_suppl)):
        feature_plot(event.cols_suppl[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = event.labels_suppl[i], w_a = w_a, w_b = w_b, filt = filt)
        
    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_rhorho": #acoangle depending on y1y2 sign
        y1y2_pos = np.array(event.cols[:,-3][filt]*event.cols[:,-2][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        feature_plot(event.cols[:,-4], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "acoangle_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        feature_plot(event.cols[:,-4], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "acoangle_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        
    if args.FEAT == "Variant-1.1" and args.TYPE == "nn_a1rho": #acoangle depending on y1y2 sign
        y1y2_pos = np.array(event.cols[:,-9][filt]*event.cols[:,-8][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        feature_plot(event.cols[:,-11], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_1_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        feature_plot(event.cols[:,-11], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_1_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-8][filt]*event.cols[:,-7][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        feature_plot(event.cols[:,-10], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_2_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        feature_plot(event.cols[:,-10], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_2_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-4][filt]*event.cols[:,-3][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        feature_plot(event.cols[:,-6], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_3_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        feature_plot(event.cols[:,-6], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_3_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)
        y1y2_pos = np.array(event.cols[:,-3][filt]*event.cols[:,-2][filt] >= 0)
        y1y2_neg = np.array([not i for i in y1y2_pos])
        feature_plot(event.cols[:,-5], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_4_y1y2_pos", w_a = w_a*y1y2_pos, w_b = w_b*y1y2_pos, filt = filt)
        feature_plot(event.cols[:,-5], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = "aco_angle_4_y1y2_neg", w_a = w_a*y1y2_neg, w_b = w_b*y1y2_neg, filt = filt)


