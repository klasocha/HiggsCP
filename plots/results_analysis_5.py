import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.metrics_utils import  calculate_deltas_signed_pi_topo
from utilities.data_utils import read_np


def draw(args):
    # Preparing the output directory
    num_classes = args.NUM_CLASSES
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_5",
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"regr_argmaxs_delt_argmax_rhorho_Variant-All_topo_nc_{num_classes}_" + \
        f"{filtered}.{args.FORMAT}"
    output_path = os.path.join(output_path, filename)

    # Loading data
    dataset = filtered + '_' + args.DATASET
    calc_argmaxs = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy"))
    preds_argmaxs = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy"))

    # Computing the needed values
    delt_argmaxs = calc_argmaxs - preds_argmaxs
    # TODO: poprawic!!! delta < 3.1415
    delt_argmaxs = calculate_deltas_signed_pi_topo(calc_argmaxs, preds_argmaxs)
    
    # k2PI = 2 * np.pi
    # calc_argmaxs= calc_argmaxs/k2PI
    mean = np.mean(delt_argmaxs)
    std  = np.std(delt_argmaxs)

    # Preparing the plot
    plt.hist(delt_argmaxs, histtype="step", bins=num_classes, color="black")
    plt.xlim([-3.2, 3.2])
    plt.xlabel(r'$\Delta \alpha^{CP}_{max}$ [rad]')
    plt.gca()
    table_vals=[[r"Regression: $\alpha^{CP}_{max}$"],
                [" "],
                [r"mean = {:0.3f} $\pm$ {:1.3f} [rad]".format(mean, 0.003)],
                ["std = {:1.3f} [rad]".format(std)]
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
    plt.savefig(output_path)
    print(f"The plot has been saved as {output_path}")

    plt.clf()