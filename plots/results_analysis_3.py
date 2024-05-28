""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for the "soft_argmaxs" model configuration """

import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.metrics_utils import  calculate_deltas_signed
from scipy import stats
from utilities.data_utils import read_np


def draw(args):
    # Preparing the output directory
    num_classes = args.NUM_CLASSES
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_3",
                              args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"soft_argmaxs_delt_rhorho_{args.FEAT}_nc_{num_classes}_" + \
        f"{filtered}"
    output_path = os.path.join(output_path, filename)

    # Loading data
    dataset = filtered + '_' + args.DATASET
    calc_hits_argmaxs = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy"))
    preds_hits_argmaxs = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy"))

    # Computing the needed values
    data_len = calc_hits_argmaxs.shape[0]
    preds_argmaxs = np.zeros((data_len, 1))
    calc_argmaxs = np.zeros((data_len, 1))

    for i in range(data_len):
        preds_argmaxs[i] = np.argmax(preds_hits_argmaxs[i])
        calc_argmaxs[i] = np.argmax(calc_hits_argmaxs[i])

    k2PI= 2 * np.pi
    delt_argmaxs =  calculate_deltas_signed(
        np.argmax(preds_hits_argmaxs[:], axis=1), 
        np.argmax(calc_hits_argmaxs[:], axis=1), num_classes)      
    delt_argmaxs_rad = delt_argmaxs * k2PI / (num_classes - 1)
 
    meanrad = np.mean(delt_argmaxs) * k2PI / (num_classes - 1)
    stdrad = np.std(delt_argmaxs) * k2PI / (num_classes - 1)
    meanerrrad = stats.sem(delt_argmaxs) * k2PI / (num_classes - 1) 
    
    # Preparing the plot
    plt.hist(delt_argmaxs_rad, histtype='step', bins=num_classes, color='black')
    plt.xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')    
    plt.gca()
    table_vals=[[r'Classification: $\alpha^{CP}_{max}$'],
                [" "],
                [r"mean = {:0.3f}$\pm$ {:1.3f} [rad]".format(meanrad, meanerrrad)],
                ["std = {:1.3f} [rad]".format(stdrad)]
                ]
    table = plt.table(cellText=table_vals,
                    colWidths = [0.40],
                    cellLoc="left",
                    loc='upper right')
    table.set_fontsize(14)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)
    plt.tight_layout()
    
    # Saving the plot
    plt.savefig(f"{output_path}.pdf")
    plt.savefig(f"{output_path}.png")
    plt.savefig(f"{output_path}.eps")
    print(f"The plot has been saved as {output_path}")

    plt.clf()