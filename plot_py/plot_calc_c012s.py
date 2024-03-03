""" 
This program generates diagrams depicting the distribution of the calculated 
(ideal, not predicted) C0/C1/C2 coefficients stored in a file named "c012s.npy" located
in the directory specified by the command line argument. 

Try to run it as the following (let us suppose the coefficients are stored in "data/c012s.npy" 
and you want to save the results in "plot_py/figures/"): 
    $ python plot_py/plot_calc_c012s.py --input "data" --output "plot_py/figures" --show True --format "png"
"""

import os, errno, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Command line arguments needed for running the program independently
parser = argparse.ArgumentParser(description='Diagram creator')
parser.add_argument("-i", "--input", dest="IN", type=Path, help="input data path", default="../temp_data")
parser.add_argument("-o", "--output", dest="OUT", type=Path, help="output path for diagrams", default="figures")
parser.add_argument("-f", "--format", dest="FORMAT", 
                    help='the format of the output diagrams ("png"/"pdf"/"eps")', default="png")
parser.add_argument("-s", "--show", dest="SHOW", type=bool, 
                    help='set to False to save the diagrams without showing them', default=True)
args = parser.parse_args()

# Reading the calculated coefficients
filename = "c012s.npy"
filepath = os.path.join(args.IN, filename)
with open(filepath, 'rb') as f:
    calc_c012s = np.load(f)


def draw_distribution(coefficients, type):
    """ Draw the distribution for a given list of coefficients """
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

    # Saving the diagram
    plt.savefig(os.path.join(output_path, f"c{type}_distribution.{args.FORMAT}"))

    # Showing the diagram
    if args.SHOW:
        plt.show()


# Generating the diagrams showing C0/C1/C2 distribution
for i in range(3):
    draw_distribution(calc_c012s[i], i)