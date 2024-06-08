"""
This program prepares the control plots of the Wt(alphaCP) distribution: 
the mean values of all the weights obtained by filtering the original weights with 
the unweighted events map (according to each of the alphaCP hypotheses).

Let us suppose the original weights calculated via the C0/C1/C2 coefficients are stored in
"data/weights_multiclass_51.npy" and the unweighted events weight values are in 
"data/unwt_multiclass_51.npy". If you want to store the plots in 
"plots/figures/unwt_weight_distribution_for_51_classes/" you can run the following command:

    $ python main.py --action "plot" --input "data" --output "plots/figures" --format "png" 
    --option "UNWEIGHTED-EVENTS-WEIGHTS" --num_classes "51"

Notice: "--show" option is also supported. Use it to show the plots as they are created. """

import numpy as np
from utilities.data_utils import read_np
import matplotlib.pyplot as plt
import os, errno

def read_np(filename):
    """ Return the data loaded from a NPY file """
    with open(filename, 'rb') as f:
        return np.load(f)

def draw(args):
    """ Plot the Wt(alphaCP) distribution: the mean values of all the weights obtained
    by filtering the original weights with the unweighted events map 
    (according to each of the alphaCP hypotheses) """
    # Getting the unweighted events weight values, as well the original weights 
    unwt_weights = read_np(os.path.join(os.path.normpath(args.IN), f"unwt_multiclass_{int(args.NUM_CLASSES)}.npy"))
    weights = read_np(os.path.join(os.path.normpath(args.IN), f"weights_multiclass_{int(args.NUM_CLASSES)}.npy"))
    alphaCP_range = np.linspace(0, 2 * np.pi, int(args.NUM_CLASSES))

    for hypothesis in range(int(args.NUM_CLASSES)):
    
        # Clearing all the weights that do not belong to the given hypothesis
        w = np.copy(weights)
        zero_mask = unwt_weights[:, hypothesis] != 1.0
        for i in range(int(args.NUM_CLASSES)):
            w[zero_mask, i] = 0.0
        w = np.mean(w, axis=0)
        
        # Drawing the Weight(alphaCP) function for the given hypothesis
        plt.clf()
        plt.scatter(alphaCP_range, w, color="black", label=(rf"""$Unweighted\ by\ {{\alpha^{{CP}}}}={
            (np.pi * 2 / (int(args.NUM_CLASSES) - 1) * hypothesis):.2f}$"""))
        plt.xlabel(r"${\alpha^{CP}} (rad)$", loc="right")
        plt.ylabel(r"wt (mean)", loc="top")
        plt.ylim([0, 1])
        plt.legend(loc="upper right")
        plt.title(f"Actual alphaCPmax = {np.argmax(w)}")
        # Creating the output folder
        output_path = os.path.join(os.path.normpath(args.OUT), f"unwt_weight_distribution_for_{int(args.NUM_CLASSES)}_classes")
        try:
            os.makedirs(output_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Saving the plot
        output_path = os.path.join(output_path, f"hypothesis_{hypothesis}.{args.FORMAT}")
        plt.savefig(output_path)
        print("The plot of the Wt(alphaCP) distribution filtered by the unweighting",
              f"map (alphaCP={hypothesis}) has been saved in {output_path}")

        # Showing the plot
        if (args.SHOW):
            plt.show()