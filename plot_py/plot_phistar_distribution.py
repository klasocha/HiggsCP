"""
This program prepares the control plots of the distribution of the phistar variable 
for various hypotheses of alphaCP. We would like to plot the distributions of these 
variables using weights for different hypotheses of alphaCP, without conditioning 
on the sign of y1*y2, and separately grouping y1*y1>0, y1*y2<0.

Let us suppose the "event" object having the attribute
responsible for storing all the features, including phistar, y1 and y2 if "Variant-1.1"
has been chosen, is stored in "data/rhorho_event.obj" and you want to save the results in "plot_py/figures/".
Then you need to run "plots.py" in the following manner (hypothesis is an alphaCP class for the
weighted distribution plots): 

    $ python plots.py --option PHISTAR-DISTRIBUTION --input "data" --output "plot_py/figures" 
      --format "png" --show --num_classes 11 --hypothesis 2 

Or for the unweighted events:

    $ python plots.py --option PHISTAR-DISTRIBUTION --input "data" --output "plot_py/figures" 
      --format "png" --show --num_classes 11 --hypothesis 2 --use_unweighted_events

Notice: hypotheses range depends on the number of classes. For example, if --num_classes=21, then
you can set --hypothesis from 0 to 20 (where 0 means 0 rad, 20 means 6.28 rad). Alternatively,
you can set the --num_classes=51 and use --hypothesis={[0, 50]}.

This program needs to be run as a module because it utilises the deserialisation mechanism used by
the pickle module, which needs to know where the RhoRhoEvent class was located when the object was
being serialised.
"""

import os, errno, pickle
import matplotlib.pyplot as plt
import numpy as np
from src_py.data_utils import read_np


def draw_distribution(variable, output_name, args, labels=None, weights=None, 
                      colors=None, xlabel=None, title=None):
    """ Draw the distribution of the given variable """
    plt.clf()
    if weights is None:
        plt.hist(variable, histtype='step', bins=50, color = 'black', label=labels)
    else:
        for v, w, l, c in zip(variable, weights, labels, colors):
            counts, bins = np.histogram(v, weights=w, bins=25)
            plt.scatter(bins[:-1], counts, marker='^', lw=1, ls='dashed',
                        label=l, c=c)
            plt.ylim(0, np.max(counts) * 2)
            plt.title(r"${\alpha^{CP}}$ = " + title)
    if labels is not None:
        plt.legend()
    plt.xlabel(xlabel, loc="right")
    plt.ylabel("Entries", loc="top")
    plt.tight_layout()

    # Creating the output folder
    output_path = os.path.join(args.OUT, f"phistar_y1y2_for_{args.NUM_CLASSES}_classes")
    output_path = os.path.join(output_path, "unweighted_events" if args.USE_UNWEIGHTED_EVENTS else "weighted_events")
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Saving the plot
    output_path = os.path.join(output_path, f"{output_name}_distribution.{args.FORMAT}")
    plt.savefig(output_path)
    print(f"Phistar distribution plot has been saved in {output_path}")

    # Showing the plot
    if args.SHOW:
        plt.show()


def draw(args):
    """ Call the draw_distribution(). If weights are given by specifying the args.HYPOTHESES,
    it passes lists of different values to the draw_distribution(). """

    # Reading the serialised "event" (RhoRhoEvent) object
    with open(os.path.join(args.IN, "rhorho_event.obj"), 'rb') as f:
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
        # Preparing the weights relevent to the given hypothesis
        if args.USE_UNWEIGHTED_EVENTS:
            weights = read_np(os.path.join(args.IN, f"unwt_multiclass_{args.NUM_CLASSES}.npy"))[:, int(args.HYPOTHESIS)]
        else:
            weights = read_np(os.path.join(args.IN, f"weights_multiclass_{args.NUM_CLASSES}.npy"))[:, int(args.HYPOTHESIS)]
        
        # Generating the plot showing phistar grouped by y1*y2 > 0 and y1*y2 < 0, 
        # the distribution is weighted by the weights values specific to the given hypothesis
        variables = [phistar_negative, phistar_positive]
        weights=[weights[y1y2_negative_mask], weights[y1y2_positive_mask]]
        alphaCP = (2 * np.pi) / (args.NUM_CLASSES - 1) * float(args.HYPOTHESIS)
        draw_distribution(variable=variables, 
                          output_name=f"phistar_y1y2_alphaCP_{args.HYPOTHESIS}_out_of_{args.NUM_CLASSES - 1}", 
                          args=args, weights=weights, colors=['black', 'red'],
                          labels=[r"${\phi* (y^+_\rho y^-_\rho < 0)}$", r"${\phi* (y^+_\rho y^-_\rho > 0)}$"],
                          xlabel=r"${\phi_{\rho \rho}}$", title="{:.2f} rad".format(alphaCP))
    else: 
        # The same but without taking into account any specific hypothesis in terms of the weights values
        draw_distribution(variable=phistar, output_name="phistar", xlabel=r"${\phi_{\rho \rho}}$", args=args)
        draw_distribution(variable=y1, output_name="y1", xlabel="${y_1}$", args=args)
        draw_distribution(variable=y2, output_name="y2", xlabel="${y_2}$", args=args)