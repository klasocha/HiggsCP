""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for "soft_weights" and "regr_weights" 
model configurations """

import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.data_utils import read_np
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed


def draw(args):
    # Preparing the output directory 
    num_classes = int(args.NUM_CLASSES)
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_1", 
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    if args.TRAINING_METHOD == "soft_weights":
        filename = f"soft_wt_delt_argmax_rhorho_{args.FEAT}_nc_{num_classes}_" + \
            f"{filtered}"
    if args.TRAINING_METHOD == "regr_weights":
        filename = f"regr_wt_delt_argmax_rhorho_{args.FEAT}_nc_{num_classes}_" + \
            f"{filtered}"
    output_path = os.path.join(output_path, filename)

    # Loading calculated and true weights
    dataset = filtered + '_' + args.DATASET
    calc_w  = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy"))
    preds_w  = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy"))

    # Computing the difference
    k2PI = 2 * np.pi
    delt_argmax = calculate_deltas_signed(np.argmax(preds_w[:], axis=1), 
                                          np.argmax(calc_w[:], axis=1), num_classes)    
    delt_argmax_rad = delt_argmax * k2PI / (num_classes - 1)

    # Preparing the plot
    bins = np.max(delt_argmax) - np.min(delt_argmax) + 1
    plt.hist(delt_argmax_rad, histtype='step', bins=bins, color='black')
    plt.xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')
    plt.ylabel('Entries')
    plt.gca()
    
    meanrad = np.mean(delt_argmax, dtype=np.float64) * k2PI / (num_classes - 1)
    stdrad  = np.std(delt_argmax, dtype=np.float64) * k2PI / (num_classes - 1)
    meanerrrad = stats.sem(delt_argmax) * k2PI / (num_classes - 1)

    if args.TRAINING_METHOD == "soft_weights":
        table_title = [r"Classification: $wt$"]
    if args.TRAINING_METHOD == "regr_weights":
        table_title = [r"Regression: $wt$"]

    table_vals=[table_title,
                [" "],
                [r"mean = {:0.3f} $\pm$ {:1.3f}[rad]".format(meanrad, meanerrrad)],
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