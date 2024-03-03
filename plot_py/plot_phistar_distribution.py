"""
This program prepares the control plots of the distribution of the phistar variable 
for various hypotheses of alphaCP. We would like to plot the distributions of these 
variables using weights for different hypotheses of alphaCP, without conditioning 
on the sign of y1*y2, and separately grouping y1*y1>0, y1*y2<0.

Try to run it as the following (let us suppose the "event" object having the attribute
responsible for storing all the features, including phistar, y1 and y2 if "Variant-1.1"
has been chosen, is stored in "data/rhorho_event.obj" and you want to save the results in "plot_py/figures/")
you need to run "plots.py" in the following manner: 
    $ python plots.py --option PHISTAR-DISTRIBUTION --input "data" --output "plot_py/figures" --format "png" --show "True"

This program needs to be run as a module because it utilises the deserialisation mechanism used by
the pickle module, which needs to know where the RhoRhoEvent class was located when the object was
being serialised.
"""

import os, errno, pickle
import matplotlib.pyplot as plt


def draw_distribution(variable, label, args):
        """ Draw the distribution of the given variable """
        plt.hist(variable, histtype='step', bins=50, color = 'black', label=f"{label} distribution")
        plt.xlabel(label)
        plt.tight_layout()

        # Creating the output folder
        output_path = args.OUT
        try:
            os.makedirs(output_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Saving the diagram
        plt.savefig(os.path.join(output_path, f"{label}_distribution.{args.FORMAT}"))

        # Showing the diagram
        if args.SHOW:
            plt.show()


def draw(args):
    # Reading the serialised "event" (RhoRhoEvent) object
    with open(os.path.join(args.IN, "rhorho_event.obj"), 'rb') as f:
        event = pickle.load(f)

    # Extracting phistar, y1 and y2
    phistar = event.cols[:, event.feature_index_dict["aco_angle"]] 
    y1 = event.cols[:, event.feature_index_dict["tau1_y"]]
    y2 = event.cols[:, event.feature_index_dict["tau2_y"]]

    # Generating the diagrams showing phistar, y1 and y2 distribution
    draw_distribution(phistar, "phistar", args)
    draw_distribution(y1, "y1", args)
    draw_distribution(y2, "y2", args)
