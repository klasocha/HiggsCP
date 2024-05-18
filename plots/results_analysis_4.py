""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for the "regr_c012s" model configuration """

import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.cpmix_utils import weight_fun
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed
from utilities.data_utils import read_np


def calc_weights(num_classes, coeffs):
    k2PI = 2* np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights


def draw(args):
    # Preparing the output directory
    num_classes = int(args.NUM_CLASSES)
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_4",
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"regr_c012s_delt_argmax_rhorho_{args.FEAT}_nc_{num_classes}_" + \
        f"{filtered}"
    output_path = os.path.join(output_path, filename)

    # Loading the coefficients
    dataset = filtered + '_' + args.DATASET
    calc_c012s = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy"))
    preds_c012s = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy"))

    # Computing the needed values    
    kPI = np.pi
    k2PI = 2 * np.pi
    calc_w = calc_weights(num_classes, calc_c012s)
    preds_w = calc_weights(num_classes, preds_c012s)
    delt_argmax = calculate_deltas_signed(np.argmax(preds_w[:], axis=1), 
                                          np.argmax(calc_w[:], axis=1), num_classes)
    delt_argmax_pi =  delt_argmax * kPI / num_classes
    meanrad = np.mean(delt_argmax) * k2PI/num_classes
    stdrad  = np.std(delt_argmax) * k2PI/num_classes
    meanerrrad = stats.sem(delt_argmax) * k2PI/num_classes

    # Preparing the plot
    plt.hist(delt_argmax_pi, histtype='step', color="black", bins=num_classes)
    plt.ylabel('Entries')
    plt.xlabel(r'$\Delta \alpha^{CP}_{max}$ [rad]')
    plt.gca()

    table_vals=[[r"Regression: $C_0, C_1, C_2$"],
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