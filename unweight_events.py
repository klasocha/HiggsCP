import os, argparse, numpy as np, matplotlib.pyplot as plt
from src_py.data_utils import read_np
from scipy import optimize
from pathlib import Path
from src_py.cpmix_utils import weight_fun

# Command line arguments needed for running the program independently
parser = argparse.ArgumentParser(description='Unweighted events generator')
parser.add_argument("-i", "--input", dest="IN", type=Path, help="input data path", default="../temp_data")
parser.add_argument("--hypothesis", dest="HYPOTHESIS", default="00", help="Hypothesis: the alphaCP class (e.g. 02)")
parser.add_argument("--option", dest="OPTION", default="UNWEIGHT-EVENTS",  help="action (UNWEIGHT-EVENTS/PREPARE-C012S)")
args = parser.parse_args()

data_path = args.IN
unweighted_events_weights_filename = "unweighted_events_weights.npy"

if args.OPTION == "UNWEIGHT-EVENTS":
    # Loading the original weight values
    original_weights = np.transpose(np.array(read_np(os.path.join(data_path, "rhorho_raw.w.npy"))))

    # Normalising the weights by scaling them by a factor of 2
    original_weights_normalised = original_weights / 2

    # Unweighting the events
    data_len = len(original_weights)
    num_hypothesis = original_weights.shape[-1]

    unweighted_events = []
    fill_with_zeroes = lambda x : 0.0

    for hypothesis in range(num_hypothesis):
        print(f"Unweighting the events according to the alphaCP={hypothesis} hypothesis", end='\r')
        temp = np.copy(original_weights_normalised)
        for event in range(data_len):
            # Monte Carlo approach
            if temp[event, hypothesis] < np.random.random():
                temp[event] = fill_with_zeroes(temp[event, :])
        unweighted_events.append(temp)

    # Saving the unweighted events matrix
    output_path = os.path.join(data_path, unweighted_events_weights_filename)
    unweighted_events = np.array(unweighted_events)
    np.save(output_path, unweighted_events)
    print(f"Weights of the unweighted events have been saved in {output_path}")

elif args.OPTION == "PREPARE-C012S":
    # Calculating the C0, C1, C2 coefficients for the given hypothesis
    unweighted_events = read_np(os.path.join(args.IN, unweighted_events_weights_filename))
    num_hypothesis = unweighted_events.shape[-1]
    data_len = len(unweighted_events[-2])
    c012s  = np.zeros((data_len, 3))
    w = read_np(os.path.join(data_path, unweighted_events_weights_filename))[int(args.HYPOTHESIS) // 2]
    x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been used by scipy.optimize.curve_fit()", end='\r')
        coeff, _ = optimize.curve_fit(weight_fun, x, w[i, :])
        c012s[i]  = coeff
    # Saving the coefficients as NPY files
    output_path = os.path.join(data_path, f'unweighted_c012s_{args.HYPOTHESIS}.npy')
    np.save(output_path, c012s)
    print(f"C0/C1/C2 for the unweighted events have been computed and saved in {output_path}")