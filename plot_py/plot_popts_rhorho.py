"""
This program generates a plot depicting the functional form of the spin weight,
as well as the spin weight discrete values. The plot helps to
test the correctness of the weights calculated via the C0/C1/C2 coefficients.
C0/C1/C2 covariance is used to show the error.

Try to run it as the following (let us suppose the coefficients are stored in "data/c012s.npy"
and the covariance values are stored in "data/ccovs.npy", so you want to save the results in "plot_py/figures/"): 
     $ python plots.py --option C012S-WEIGHT --input "data" --output "plot_py/figures" --format "png" --show
"""
import os, errno
import numpy as np
import matplotlib.pyplot as plt
from src_py.data_utils import read_np
from src_py.cpmix_utils import weight_fun


def draw_weights_to_compare(c012s, ccovs, discrete_weights, event_index, args):
    plt.clf()
    # Drawing the true values (generated with an algorithm using Monte Carlo methods)
    x_weights = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * np.pi
    plt.scatter(x_weights, discrete_weights[:, event_index], label="Generated")
    
    # Drawing the values computed with the help of the C0/C1/C2 coefficients
    # Notice: we do not use "weights_#.npy", although it already has the values computed via
    # the coefficients, as they are discrete. We need a continuous range to draw it ideally on the plot
    x_fit = np.linspace(0, 2 * np.pi)
    plt.plot(x_fit, weight_fun(x_fit, *c012s[event_index]), 
            label=f"Function \nError: {np.sqrt(np.diag(ccovs[event_index]))}", color="orange")
    
    # Configuring the plot
    plt.ylim([0.0, 2.5])
    plt.xlabel(r'$\alpha^{CP}$ [rad]')
    plt.ylabel('wt')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Creating the output folder
    output_path = args.OUT
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Showing and saving the plot
    output_path = os.path.join(args.OUT, f"calc_vs_gen_weights_event_{event_index}.{args.FORMAT}")
    plt.savefig(output_path)
    print(f"The plot showing the correctness of the calculated C0/C1/C2 has been saved in {output_path}")

    if args.SHOW:
        plt.show()


def draw(args):
    """ Draw the computed values and the true values of the weight(alphaCP) function """

    # Reading the calculated coefficients, covariances, as well as the calculated weights
    c012s_path = os.path.join(args.IN, "c012s.npy")
    ccovs_path = os.path.join(args.IN, "ccovs.npy")
    discrete_weights_path = os.path.join(args.IN, "rhorho_raw.w.npy")

    c012s = read_np(c012s_path)
    ccovs = read_np(ccovs_path)
    discrete_weights = read_np(discrete_weights_path)

    # Calling the main drawing function for some sample events
    draw_weights_to_compare(c012s, ccovs, discrete_weights, 0, args)
    draw_weights_to_compare(c012s, ccovs, discrete_weights, 10, args)
    draw_weights_to_compare(c012s, ccovs, discrete_weights, 1000, args)
    draw_weights_to_compare(c012s, ccovs, discrete_weights, 2000, args)