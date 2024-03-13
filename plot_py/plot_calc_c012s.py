""" 
This program generates the plots depicting the distribution of the calculated 
(ideal, not predicted) C0/C1/C2 coefficients stored in a file named "c012s.npy" located
in the directory specified by the command line argument. 

Try to run it as the following (let us suppose the coefficients are stored in "data/c012s.npy" 
and you want to save the results in "plot_py/figures/"): 

    $ python plots.py --option C012S-DISTRIBUTION --input "data" --output "plot_py/figures" --format "png" --show "True"
"""

import os, errno
import numpy as np
import matplotlib.pyplot as plt


def draw_distribution(calc_c012s, type, args):
    plt.hist(calc_c012s[:, type], histtype='step', bins=50,  color = 'black')
    plt.xlabel(f"$\mathregular{{C_{type}}}$")
    plt.tight_layout()

    # Creating the output folder
    output_path = args.OUT
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Saving the plot
    plt.savefig(os.path.join(output_path, f"c{type}_distribution.{args.FORMAT}"))

    # Showing the plot
    if args.SHOW:
        plt.show()


def draw(args):
    """ Draw the distribution for a given list of coefficients """
    # Reading the calculated coefficients
    filename = "c012s.npy"
    filepath = os.path.join(args.IN, filename)
    with open(filepath, 'rb') as f:
        calc_c012s = np.load(f)

    # Generating the plot showing C0/C1/C2 distribution
    for i in range(3):
        draw_distribution(calc_c012s, i, args)