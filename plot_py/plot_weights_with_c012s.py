"""
This program prepares a plot of the function Wt(alphaCP) using the weights calculated 
with the use of C0/C1/C2 as well as C1/C2 (without C0) 
Try to run it as the following (let us suppose the coefficients are stored in "data/c012s.npy"
and you want to save the resul in "plot_py/figures/"):

    $ python plots.py --option WEIGHTS-FOR-EVENT-VIA-C012 --input "data" 
      --output "plot_py/figures" --format "png" --show 

    BETA-VERSION: if --use_unweighted_events:
    ======================================================================================
    $ python plots.py --option WEIGHTS-FOR-EVENT-VIA-C012 --input "data" 
      --output "plot_py/figures" --format "png" --show --use_unweighted_events 
      --hypothesis 00
    ======================================================================================
"""

import numpy as np
from src_py.data_utils import read_np
from src_py.cpmix_utils import weight_fun
import matplotlib.pyplot as plt
import os, errno


def draw(args):
    """ Plot the function Wt(alphaCP) using the weights calculated with the use of 
    C0/C1/C2 as well as C1/C2 (without C0) """
   
    # Discretisation level
    n_discrete_values = 50

    # Sample events indices
    sample_events = [4, 2569, 8533, 55, 995]

    if args.USE_UNWEIGHTED_EVENTS:
        # Reading the unweighted events coefficients for a given hypotheses
        c012s = read_np(os.path.join(args.IN, f"unweighted_c012s_{args.HYPOTHESIS}.npy"))
        c012s = np.mean(c012s, axis=0)
        # Shape modifications for backward compatibility (use_unweighted_events option was added later):
        sample_events = [0]
        c012s = np.array([c012s]) * 20
    else:
        # Reading the initial events coefficients
        c012s = read_np(os.path.join(args.IN, "c012s.npy"))

    alphaCP_range = np.linspace(0, 2 * np.pi, n_discrete_values)
    calculated_weights = []
    for i in sample_events:
        calculated_weights.append(weight_fun(alphaCP_range, *c012s[i]))
    draw_distribution(calculated_weights, c012s[sample_events, 0], alphaCP_range, args)


def draw_distribution(weights, c0, alphaCP_range, args):
    _, axs = plt.subplots(1, 2, figsize=(15, 4.5))
    markers = ['s', '*', '.', 'v', '^']
    
    for i in range(len(weights)):
        # Weights calculated via C0/C1/C2
        ax = axs[0]
        if not args.USE_UNWEIGHTED_EVENTS:
            ax.set_ylim(0, 2)
            label = f"Event {i + 1}"
            color = None
        else:
            label = f"Based on the unweighted \nevents for alphaCP = {args.HYPOTHESIS}"
            color = "black"
        ax.scatter(alphaCP_range, weights[i], s=19, marker=markers[i], label=label, color=color)
        ax.legend(loc='lower right')
        ax.set_ylabel(r"$C_0 + C_1 cos({\alpha}^{CP}) + C_2 sin({\alpha^{CP}})$", loc="top")
        ax.set_xlabel(r"${\alpha^{CP}}$ [rad]", loc="right")
        
        # Weights calculated via C1/C2
        ax = axs[1]
        if not args.USE_UNWEIGHTED_EVENTS:
            ax.set_ylim(-1, 1)
            label = f"Event {i + 1}"
            color = None
        else:
            color = "black"
            label = f"Based on the unweighted \nevents for alphaCP={args.HYPOTHESIS}"
        ax.scatter(alphaCP_range, weights[i] - c0[i], s=19, marker=markers[i], label=label, color=color)
        ax.legend(loc='lower right')
        ax.set_ylabel(r"$C_1 cos({\alpha}^{CP}) + C_2 sin({\alpha^{CP}})$", loc="top")
        ax.set_xlabel(r"${\alpha^{CP}}$ [rad]", loc="right")

    plt.setp(axs, xlim=(0, alphaCP_range[-1]), title=r"${\alpha^{CP}}$", xlabel=r"${\alpha^{CP}}$ [rad]")

    # Creating the output folder
    output_path = args.OUT
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Saving the diagram
    if args.USE_UNWEIGHTED_EVENTS:
        output_path = os.path.join(output_path, f"unweighted_c012s_vs_c12s_{args.HYPOTHESIS}.{args.FORMAT}")
    else:
        output_path = os.path.join(output_path, f"weights_c012s_vs_c12s.{args.FORMAT}")
    
    plt.savefig(output_path)
    print(f"Wt(alphaCP) subplots have been saved in {output_path}")

    # Showing the diagram
    if args.SHOW:
        plt.show()
