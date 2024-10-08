""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for "soft_weights" and "regr_weights" 
model configurations.

EXAMPLE 1
Making a plot for one feature set:
    --input "results/regr_weights/51_classes_variant_all/predictions"
    --features "Variant-All"

EXAMPLE 2
Making a plot for several feature sets (up to six):
    --input "results/regr_weights/51_classes_variant_"
    --features "Variant-All-4.1-1.1"

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.data_utils import read_np
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed


def draw(args):
    # Parsing chosen feature sets (e.g. --features "Variant-All-1.1")
    feature_list = args.FEAT.split('-')
    feature_list = [feature_list[x] for x in range(1, len(feature_list))]
    features = "Variant"
    for feature in feature_list:
        features += '-' + feature 

    # Preparing the output directory 
    num_classes = int(args.NUM_CLASSES)
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_1", 
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    if args.TRAINING_METHOD == "soft_weights":
        filename = f"soft_wt_delt_argmax_rhorho_{features}_nc_{num_classes}_" + \
            f"{filtered}"
    if args.TRAINING_METHOD == "regr_weights":
        filename = f"regr_wt_delt_argmax_rhorho_{features}_nc_{num_classes}_" + \
            f"{filtered}"
    output_path = os.path.join(output_path, filename)

    # Loading calculated and true weights
    dataset = filtered + '_' + args.DATASET
    calc_ws = []
    preds_ws = []
    if len(feature_list) == 1:
        input_path = os.path.normpath(args.IN)
        calc_ws.append(read_np(os.path.join(input_path, f"{dataset}_calc.npy")))
        preds_ws.append(read_np(os.path.join(input_path, f"{dataset}_preds.npy")))
    else:
        for feature in feature_list:
            input_path = os.path.join(os.path.normpath(args.IN + feature.lower()), "predictions")
            calc_ws.append(read_np(os.path.join(input_path, f"{dataset}_calc.npy")))
            preds_ws.append(read_np(os.path.join(input_path, f"{dataset}_preds.npy")))   

    # Computing the difference
    k2PI = 2 * np.pi
    delt_argmaxs, delt_argmax_rads, meanrads, stdrads, meanraderrs = [], [], [], [], []
    for i in range(len(feature_list)):
        delt_argmaxs.append(calculate_deltas_signed(np.argmax(preds_ws[i][:], axis=1), 
                                np.argmax(calc_ws[i][:], axis=1), num_classes))    
        delt_argmax_rads.append(delt_argmaxs[i] * k2PI / (num_classes - 1))
        meanrads.append(np.mean(delt_argmaxs[i], dtype=np.float64) * k2PI / (num_classes - 1))
        stdrads.append(np.std(delt_argmaxs[i], dtype=np.float64) * k2PI / (num_classes - 1))
        meanraderrs.append(stats.sem(delt_argmaxs[i]) * k2PI / (num_classes - 1))

    # Preparing the plot
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 6])
    fig.set_size_inches(7, 6)
    colors = ['black', 'blue', 'red', 'green', 'violet', 'yellow']
    
    if len(feature_list) == 1:
        ax2.set_ylabel('Entries')
        density = False
    else:
        ax2.set_ylabel('Normalized to U.I.')
        density = True
        ax2.set_ylim([0, 4.6])
    
    for i in range(len(feature_list)):
        bins = np.max(delt_argmaxs[i]) - np.min(delt_argmaxs[i]) + 1
        ax2.hist(delt_argmax_rads[i], density=density, histtype='step', bins=bins, 
                 color=colors[i], label=f"Variant-{feature_list[i]}")
    ax2.legend()    
    
    ax2.set_xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')

    if args.TRAINING_METHOD == "soft_weights":
        table_title = [r"Classification: $wt$ (Variant-" + "/".join([feature for 
                                                                     feature in feature_list]) + ")"]
    if args.TRAINING_METHOD == "regr_weights":
        table_title = [r"Regression: $wt$ (Variant-" + "/".join([feature for 
                                                                 feature in feature_list]) + ")"]
    
    table_vals=[
        table_title,
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