""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for the "regr_c012s" model configuration 

EXAMPLE 1
Making a plot for one feature set:
    --input "results/regr_c012s/51_classes_variant_all/predictions"
    --features "Variant-All"

EXAMPLE 2
Making a plot for several feature sets (up to six):
    --input "results/regr_c012s/51_classes_variant_"
    --features "Variant-All-4.1-1.1"

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.cpmix_utils import weight_fun
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed
from utilities.data_utils import read_np


def calc_weights(num_classes, coeffs):
    x = np.linspace(0, 2 * np.pi, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights


def draw(args):
    # Parsing chosen feature sets (e.g. --features "Variant-All-1.1")
    feature_list = args.FEAT.split('-')
    feature_list = [feature_list[x] for x in range(1, len(feature_list))]
    features = "Variant"
    for feature in feature_list:
        features += '-' + feature 

    # Preparing the output directory
    num_classes = int(args.NUM_CLASSES)
    discr_level = int(args.NBINS) if args.NBINS else int(args.NUM_CLASSES)

    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_4",
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"regr_c012s_delt_argmax_rhorho_{features}_nc_{num_classes}_" + \
        f"{filtered}_bins_{discr_level}"
    output_path = os.path.join(output_path, filename)

    # Loading the coefficients
    dataset = filtered + '_' + args.DATASET
    calc_c012s, preds_c012s = [], []
    if len(feature_list) == 1:
        calc_c012s.append(read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy")))
        preds_c012s.append(read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy")))
    else:
        for feature in feature_list:
            input_path = os.path.join(os.path.normpath(args.IN + feature.lower()), "predictions")
            calc_c012s.append(read_np(os.path.join(input_path, f"{dataset}_calc.npy")))
            preds_c012s.append(read_np(os.path.join(input_path, f"{dataset}_preds.npy")))

    # Computing the needed values
    k2PI = 2 * np.pi
    delt_argmaxs, delt_argmax_rads, meanrads, stdrads, meanraderrs = [], [], [], [], []
    calc_ws, preds_ws = [], []
    for i in range(len(feature_list)):
        calc_ws.append(calc_weights(discr_level, calc_c012s[i]))
        preds_ws.append(calc_weights(discr_level, preds_c012s[i]))
        delt_argmaxs.append(calculate_deltas_signed(np.argmax(preds_ws[i][:], axis=1), 
                                            np.argmax(calc_ws[i][:], axis=1), discr_level))
        delt_argmax_rads.append(delt_argmaxs[i] * k2PI / (discr_level - 1))
        meanrads.append(np.mean(delt_argmaxs[i]) * k2PI / (discr_level - 1))
        stdrads.append(np.std(delt_argmaxs[i]) * k2PI / (discr_level - 1))
        meanraderrs.append(stats.sem(delt_argmaxs[i]) * k2PI / (discr_level - 1))

    # Preparing the plot
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 6])
    fig.set_size_inches(7, 6)
    colors = ['black', 'blue', 'red', 'green', 'violet', 'yellow']
    
    if len(feature_list) == 1:
        ax2.set_ylabel('Entries')
        density = False
    else:
        ax2.set_ylabel('Probability Density')
        density = True
    
    for i in range(len(feature_list)):
        bins = np.max(delt_argmaxs[i]) - np.min(delt_argmaxs[i]) + 1
        ax2.hist(delt_argmax_rads[i], density=density, histtype='step', bins=bins, 
                 color=colors[i], label=f"Variant-{feature_list[i]}")
    ax2.legend()
    
    ax2.set_xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')

    table_vals=[
            [r"Regression: $C_0, C_1, C_2$ (Variant-" + "/".join(
                [feature for feature in feature_list]) + ")"],
            [" "],
            ["mean = " + 
            " | ".join([r"{:0.3f} $\pm$ {:0.3f}".format(meanrad, meanraderr) for 
                                meanrad, meanraderr in zip(meanrads, meanraderrs)]) + " [rad]"], 
            ["std = " + 
            " | ".join(["{:0.3f}".format(stdrad) for stdrad in stdrads]) + " [rad]"]
        ]

    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=table_vals, colWidths = [1.0],
                        cellLoc="left", loc='upper left')
    table.set_fontsize(12)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)
    plt.tight_layout()

    # Saving the plot
    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{output_path}.{format}")
    print(f"The plot has been saved as {output_path}")

    plt.clf()