"""
This program prepares the control plots of the distribution of the phistar variable 
for various hypotheses of alphaCP. We would like to plot the distributions of these 
variables using weights for different hypotheses of alphaCP, without conditioning 
on the sign of y1*y2, and separately grouping y1*y1>0, y1*y2<0.

Let us suppose the "event" object having the attribute
responsible for storing all the features, including phistar, y1 and y2 if "Variant-1.1"
has been chosen, is stored in "data/rhorho_event_Variant-1.1.obj" and you want to save the results in "plots/figures/".
Then you need to run "plots.py" in the following manner (hypothesis is an alphaCP class for the
weighted distribution plots): 

    $ python main.py --action "plot" --option "PHISTAR-DISTRIBUTION" --input "data" --output "plots/figures" 
    --format "png" --show --num_classes "51" --feature "Variant-1.1"

Or for the unweighted events:

    $ python main.py --action "plot" --option PHISTAR-DISTRIBUTION --input "data" --output "plots/figures" 
    --format "png" --show --num_classes 51 --feature "Variant-1.1" --use_unweighted_events

You can also plot phistar distribution filtered by the given hypotheses:

    $ python main.py --action "plot" --option PHISTAR-DISTRIBUTION --input "data" --output "plots/figures" 
    --format "png" --show --num_classes 51 --feature "Variant-1.1" --hypothesis "0-4-46" 

Notice: hypotheses range depends on the number of classes. For example, if --num_classes=21, then
you can set --hypothesis from 0 to 20 (where 0 means 0 rad, 20 means 6.28 rad). Alternatively,
you can set the --num_classes=51 and use --hypothesis={[0, 50]}.

This program can only be run as a module because it utilises the deserialisation mechanism used by
the pickle module, which needs to know where the RhoRhoEvent class was located when the object was
being serialised.
"""

import os, pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from utilities.data_utils import read_np


def draw_distribution(variable, output_name, args, labels=None, weights=None, 
                      colors=None, xlabel=None, title=None):
    """ Draw the distribution of the given variable """
    plt.clf()
    if weights is None:
        plt.hist(variable, histtype='step', bins=50, color = 'black', label=labels)
    else:
        markers = ['o', '^', 'v']
        for v, w, l, c, i in zip(variable, weights, labels, colors, range(0, len(markers))):
            counts, bins = np.histogram(v, weights=w, bins=25)
            plt.scatter(bins[:-1], counts, marker=markers[i], lw=1, ls='dashed',
                        label=l, c=c)
            plt.ylim(0, np.max(counts) * 2)
            plt.title(r"${\alpha^{CP}}$ = " + title)
    if labels is not None:
        plt.legend()
    plt.xlabel(xlabel, loc="right")
    plt.ylabel("Entries", loc="top")
    plt.tight_layout()

    # Creating the output folder
    output_path = os.path.join(os.path.normpath(args.OUT), "phistar_dist", 
                               f"phistar_y1y2_for_{args.NUM_CLASSES}_classes")
    output_path = os.path.join(os.path.normpath(output_path), 
                               "unweighted_events" if args.USE_UNWEIGHTED_EVENTS else "weighted_events")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Saving the plot
    output_path = os.path.join(os.path.normpath(output_path), f"{output_name}_distribution")
    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{output_path}.{format}")
    print(f"The plot has been saved as {output_path}")

    # Showing the plot
    if args.SHOW:
        plt.show()


def draw_mult_dist(phistar, hypotheses, args, titles, weights, colors, alphaCP):
    """ Draw phistar distribution (multiple hypotheses) """
    fig, axs = plt.subplots(2, 2, figsize=(16, 7),
                            gridspec_kw={'height_ratios': [5, 1]})


    # Left subplot
    relative_amplitude_neg = []
    markers = ['o', '^', 'v']
    for i in range(len(hypotheses)):
        counts, bins = np.histogram(phistar[0], weights=weights[0][i], bins=25)
        axs[0, 0].scatter(bins[:-1], counts, marker=markers[i % len(markers)], 
                          lw=1, ls='dashed', c=colors[i % len(colors)],
                    label=r"${\alpha^{CP}}$ = " + f"{alphaCP[i]} [rad]")
        
        max_bin = counts.max()
        min_bin = counts.min()
        relative_amplitude_neg.append(np.round(
            2 * (max_bin - min_bin) / (max_bin + min_bin),
            2))
    
    axs[0, 0].set_title(titles[0])
    axs[0, 0].set_ylim(np.min(counts) / 2, np.max(counts) * 1.5)
    axs[0, 0].legend()
    axs[0, 0].set_xlabel(r"${\phi{*}}$", loc="right")
    axs[0, 0].set_ylabel("Entries", loc="top")
    axs[0, 0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axs[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Right subplot
    relative_amplitude_pos = []
    for i in range(len(hypotheses)):
        counts, bins = np.histogram(phistar[1], weights=weights[1][i], bins=25)
        axs[0, 1].scatter(bins[:-1], counts, marker=markers[i % len(markers)], 
                          lw=1, ls='dashed', c=colors[i % len(colors)],
                    label=r"${\alpha^{CP}}$ = " + f"{alphaCP[i]} [rad]")
        
        max_bin = counts.max()
        min_bin = counts.min()
        relative_amplitude_pos.append(np.round(
            2 * (max_bin - min_bin) / (max_bin + min_bin),
            2))
        
    axs[0, 1].set_title(titles[1])
    axs[0, 1].set_ylim(np.min(counts) / 2, np.max(counts) * 1.5)
    axs[0, 1].legend()
    axs[0, 1].set_xlabel(r"${\phi{*}}$", loc="right")
    axs[0, 1].set_ylabel("Entries", loc="top")
    axs[0, 1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axs[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Left info table
    table_vals=[
        ["Relative amplitude: " + ", ".join(f"{num}" for num in relative_amplitude_neg)]
    ]
    axs[1, 0].axis('off')
    axs[1, 0].axis('tight')
    table = axs[1, 0].table(cellText=table_vals, colWidths=[1.0],
                        cellLoc="left", loc='upper left')
    table.set_fontsize(12)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)
    
    # Right info table
    table_vals=[
         ["Relative amplitude: " + ", ".join(f"{num}" for num in relative_amplitude_pos)]
    ]
    axs[1, 1].axis('off')
    axs[1, 1].axis('tight')
    table = axs[1, 1].table(cellText=table_vals, colWidths=[1.0],
                        cellLoc="left", loc='upper left')
    table.set_fontsize(12)
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)

    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    # Showing and saving the plot
    output_path = os.path.join(os.path.normpath(args.OUT), "phistar_dist", 
                               f"phistar_y1y2_multiple_hypotheses")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 
        "phistar_y1y2_alphaCP_" + "_".join(f"{hyp}" for hyp in hypotheses) + \
            f"_out_of_{int(args.NUM_CLASSES) - 1}")
    
    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{output_path}.{format}")
    print(f"The plot has been saved as {output_path}")

    # Showing the plot
    if args.SHOW:
        plt.show()

    plt.clf()


def draw(args):
    """ Call the draw_distribution(). If weights are given by specifying the args.HYPOTHESES,
    it passes lists of different values to the draw_distribution(). """

    # Parsing chosen hypotheses (e.g. --hypothesis "0-5-23")
    if args.HYPOTHESIS != "None":
        hypotheses = args.HYPOTHESIS.split('-')
        hypotheses = [int(hyp) for hyp in hypotheses]

    # Reading the serialised "event" (RhoRhoEvent) object
    with open(os.path.join(os.path.normpath(args.IN), 
                           f"rhorho_event_{args.FEAT}.obj"), 'rb') as f:
        event = pickle.load(f)

    # Extracting phistar, y1 and y2
    phistar = event.cols[:, event.feature_index_dict["aco_angle"]] 
    y1 = event.cols[:, event.feature_index_dict["tau1_y"]]
    y2 = event.cols[:, event.feature_index_dict["tau2_y"]]
    
    # Preparing phistart for y1*y2 > 0 and y1*y2 < 0
    y1y2_positive_mask = y1 * y2 > 0
    y1y2_negative_mask = y1 * y2 < 0
    phistar_positive = phistar[y1y2_positive_mask] 
    phistar_negative = phistar[y1y2_negative_mask]
    
    # Loading weights if a hypothesis (alphaCP class) has been provided for the distribution 
    if args.HYPOTHESIS != "None":
        
        # Preparing the weights relevant to the given hypothesis
        weights, weights_pos, weights_neg, alphaCP = [], [], [], []
        for hyp, i in zip(hypotheses, range(0, len(hypotheses))):
            if args.USE_UNWEIGHTED_EVENTS:
                weights.append(read_np(os.path.join(os.path.normpath(args.IN), 
                            f"unwt_multiclass_{args.NUM_CLASSES}.npy"))[:, hyp])
            else:
                weights.append(read_np(os.path.join(os.path.normpath(args.IN), 
                            f"weights_multiclass_{args.NUM_CLASSES}.npy"))[:, hyp])
        
            # Generating the plot showing phistar grouped by y1*y2 > 0 and y1*y2 < 0, 
            # the distribution is weighted by the weights values specific to the given hypothesis
            weights_pos.append(weights[i][y1y2_positive_mask])
            weights_neg.append(weights[i][y1y2_negative_mask])

            alphaCP.append(np.round((2 * np.pi) / (int(args.NUM_CLASSES) - 1) * hyp, 2))
        draw_mult_dist(phistar=[phistar_negative, phistar_positive], 
                    hypotheses=hypotheses, args=args, weights=[weights_neg, weights_pos],
                    colors=['black', 'red', 'blue'],
                    titles=[r"${\phi* (y^+_\rho y^-_\rho < 0)}$", r"${\phi* (y^+_\rho y^-_\rho > 0)}$"],
                    alphaCP=alphaCP)
    else: 
        # The same but without taking into account any specific hypothesis in terms of the weights values
        draw_distribution(variable=phistar, output_name="phistar", xlabel=r"${\phi{*}}$", args=args)
        draw_distribution(variable=y1, output_name="y1", xlabel="${y_1}$", args=args)
        draw_distribution(variable=y2, output_name="y2", xlabel="${y_2}$", args=args)