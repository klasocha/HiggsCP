"""
This program generates a diagram depicting the functional form of the spin weight,
as well as the spin weight discrete values. The diagram helps to
test the correctness of the weights calculated via the C0/C1/C2 coefficients.
C0/C1/C2 covariance is used to show the error.

Try to run it as the following (let us suppose the coefficients are stored in "data/c012s.npy"
and the covariance values are stored in "data/ccovs.npy", so you want to save the results in "plot_py/figures/"): 
    $ python plot_py/plot_popts_rhorho.py --input "data" --output "plot_py/figures" --show True --format "png"
"""
import argparse, os
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


def read_np(filepath):
    """ Load and return values from a given NPY file """
    with open(filepath, 'rb') as f:
        values = np.load(f)
    return values


def weight_fun(x, a, b, c):
    """ Compute the functional form of the spin weight """
    return a + b * np.cos(x) + c * np.sin(x)

# Reading the calculated coefficients, covariances, as well as the calculated weights
c012s_path = os.path.join(args.IN, "c012s.npy")
ccovs_path = os.path.join(args.IN, "ccovs.npy")
discrete_weights_path = os.path.join(args.IN, "rhorho_raw.w.npy")

c012s = read_np(c012s_path)
ccovs = read_np(ccovs_path)
discrete_weights = read_np(discrete_weights_path)


def draw_weights(c012s, ccovs, discrete_weights, event_index):
    """ Draw the computed values and the true values of the weight(alphaCP) function """

    # Drawing the true values (generated with an algorithm using Monte Carlo methods)
    x_weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * np.pi
    plt.scatter(x_weights, discrete_weights[:, event_index], label="Generated")
    
    # Drawing the values computed with the help of the C0/C1/C2 coefficients
    # Notice: we do not use "weights.npy", although it already has the values computed via
    # the coefficients, as they are discrete. We need a continuous range to draw it ideally on the diagram
    x_fit = np.linspace(0, 2 * np.pi)
    plt.plot(x_fit, weight_fun(x_fit, *c012s[event_index]), 
             label=f"Function \nError: {np.sqrt(np.diag(ccovs[event_index]))}", color="orange")
    
    # Configuring the diagram
    plt.ylim([0.0, 2.5])
    plt.xlabel(r'$\alpha^{CP}$ [rad]')
    plt.ylabel('wt')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Showing and saving the diagrams
    if args.SHOW:
        plt.show()
    plt.savefig(os.path.join(args.OUT, f"calc_vs_gen_weights_event_{event_index}.{args.FORMAT}"))


# Calling the main drawing function for some sample events
draw_weights(c012s, ccovs, discrete_weights, 0)
draw_weights(c012s, ccovs, discrete_weights, 10)
draw_weights(c012s, ccovs, discrete_weights, 1000)
draw_weights(c012s, ccovs, discrete_weights, 2000)

# TODO: find out what the purpose of the below lines was 
# Comment from ERW
# ax = plt.gca()
# what is wrong with line below?
# ax.annotate("chi2/Ndof = {%0.3f}\n".format(chi2), xy=(0.7, 0.85), xycoords='axes fraction', fontsize=12)