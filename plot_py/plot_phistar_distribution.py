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
      --format "png" --show --hypothesis 02

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
    output_path = args.OUT
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Saving the diagram
    output_path = os.path.join(output_path, f"{output_name}_distribution.{args.FORMAT}")
    plt.savefig(output_path)
    print(f"Distribution plot has been saved in {output_path}")

    # Showing the diagram
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
        # Preparing the weights
        weights = read_np(os.path.join(args.IN, f"rhorho_raw.w_{args.HYPOTHESIS}.npy"))
        
        # Generating the plot showing phistar grouped by y1*y2 > 0 and y1*y2 < 0, 
        # the distribution is weighted by the weights values specific to the given hypothesis
        variables = [phistar_negative, phistar_positive]
        weights=[weights[y1y2_negative_mask], weights[y1y2_positive_mask]]
        alphaCP = float(args.HYPOTHESIS) / 10 * np.pi
        draw_distribution(variable=variables, 
                          output_name=f"phistar_y1y2_alphaCP_{args.HYPOTHESIS}", 
                          args=args, weights=weights, colors=['black', 'red'],
                          labels=[r"${\phi* (y^+_\rho y^-_\rho < 0)}$", r"${\phi* (y^+_\rho y^-_\rho > 0)}$"],
                          xlabel=r"${\phi_{\rho \rho}}$", title="{:.2f} rad".format(alphaCP))
    else: 
        # The same but without taking into account any specific hypothesis in terms of the weights values
        draw_distribution(variable=phistar, output_name="phistar", xlabel=r"${\phi_{\rho \rho}}$", args=args)
        draw_distribution(variable=y1, output_name="y1", xlabel="${y_1}$", args=args)
        draw_distribution(variable=y2, output_name="y2", xlabel="${y_2}$", args=args)