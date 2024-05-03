import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.data_utils import read_np
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed


def draw(args):
    # Preparing the output directory 
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_1")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = f"soft_wt_delt_argmax_rhorho_Variant-All_nc_{args.NUM_CLASSES}.{args.FORMAT}"
    output_path = os.path.join(output_path, filename)

    # Loading calculated and true weights
    calc_w  = read_np(os.path.join(os.path.normpath(args.IN), 'test_calc.npy'))
    preds_w  = read_np(os.path.join(os.path.normpath(args.IN), 'test_preds.npy'))

    # Computing the difference
    num_classes = int(args.NUM_CLASSES)
    delt_argmax =  calculate_deltas_signed(np.argmax(preds_w[:], axis=1), 
                                                np.argmax(calc_w[:], axis=1), num_classes)      

    # Preparing the plot
    plt.hist(delt_argmax, histtype='step', bins=num_classes, color='black')
    plt.xlabel(r'$\Delta_{class}$ [idx]')
    plt.ylabel('Entries')
    plt.gca()
    
    mean = np.mean(delt_argmax, dtype=np.float64)
    std  = np.std(delt_argmax, dtype=np.float64)
    meanerr = stats.sem(delt_argmax)
    meanrad = np.mean(delt_argmax, dtype=np.float64) * 6.28/num_classes
    stdrad  = np.std(delt_argmax, dtype=np.float64) * 6.28/num_classes
    meanerrrad = stats.sem(delt_argmax) * 6.28/num_classes

    table_vals=[[r"Classification: $wt$"],
                [" "],
                [r"mean = {:0.3f} $\pm$ {:1.3f}[idx] ".format(mean, meanerr)],
                ["std = {:1.3f} [idx]".format(std)],
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
    plt.savefig(output_path)
    print(f"The plot has been saved as {output_path}")

    plt.clf()