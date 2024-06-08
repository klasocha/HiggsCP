""" This program prepares a plot showing the mean and std of the difference
between true and predicted values for the "soft_c012s" model configuration """

import os
import matplotlib.pyplot as plt
import numpy as np
from utilities.cpmix_utils import weight_fun
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed
from utilities.data_utils import read_np


def calc_weights(num_classes, coeffs):
    x = np.linspace(0, 2 * np.pi, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights


def draw(args):
    # Preparing the output directory
    num_classes = int(args.NUM_CLASSES)
    discr_level = int(args.NBINS) if args.NBINS else int(args.NUM_CLASSES)
        
    output_path = os.path.join(os.path.normpath(args.OUT), "results_analysis_2",
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"soft_c012s_delt_argmax_rhorho_{args.FEAT}_nc_{num_classes}_" + \
        f"{filtered}_bins_{discr_level}"
    output_path = os.path.join(output_path, filename)

    # Loading the coefficients
    dataset = filtered + '_' + args.DATASET
    c0_input_path = f"{args.IN}0"
    calc_hits_c0s = read_np(os.path.join(
        os.path.normpath(c0_input_path), "predictions", f"{dataset}_calc.npy")) 
    preds_hits_c0s = read_np(os.path.join(
        os.path.normpath(c0_input_path), "predictions", f"{dataset}_preds.npy")) 

    c1_input_path = f"{args.IN}1"
    calc_hits_c1s = read_np(os.path.join(
        os.path.normpath(c1_input_path), "predictions", f"{dataset}_calc.npy")) 
    preds_hits_c1s = read_np(os.path.join(
        os.path.normpath(c1_input_path), "predictions", f"{dataset}_preds.npy")) 

    c2_input_path = f"{args.IN}2"
    calc_hits_c2s = read_np(os.path.join(
        os.path.normpath(c2_input_path), "predictions", f"{dataset}_calc.npy")) 
    preds_hits_c2s = read_np(os.path.join(
        os.path.normpath(c2_input_path), "predictions", f"{dataset}_preds.npy")) 
        
    # Computing the needed values
    data_len = calc_hits_c0s.shape[0]

    preds_c0s = np.zeros((data_len, 1))
    calc_c0s = np.zeros((data_len, 1))
    
    preds_c1s = np.zeros((data_len, 1))
    calc_c1s = np.zeros((data_len, 1))
    
    preds_c2s = np.zeros((data_len, 1))
    calc_c2s = np.zeros((data_len, 1))

    for i in range(data_len):
        preds_c0s[i] = np.argmax(preds_hits_c0s[i])
        calc_c0s[i] = np.argmax(calc_hits_c0s[i])
    
        preds_c1s[i] = np.argmax(preds_hits_c1s[i])
        calc_c1s[i] = np.argmax(calc_hits_c1s[i])
    
        preds_c2s[i] = np.argmax(preds_hits_c2s[i])
        calc_c2s[i] = np.argmax(calc_hits_c2s[i])    

    calc_c012s = np.zeros((data_len, 3))
    preds_c012s = np.zeros((data_len, 3))

    for i in range(data_len):
        calc_c012s[i][0] = calc_c0s[i] * (2. / (num_classes - 1))
        calc_c012s[i][1] = calc_c1s[i] * (2. / (num_classes - 1)) - 1.0
        calc_c012s[i][2] = calc_c2s[i] * (2. / (num_classes - 1)) - 1.0

        preds_c012s[i][0] = preds_c0s[i] * (2. / (num_classes - 1))
        preds_c012s[i][1] = preds_c1s[i] * (2. / (num_classes - 1)) - 1.0
        preds_c012s[i][2] = preds_c2s[i] * (2. / (num_classes - 1)) - 1.0

    k2PI = 2 * np.pi
    calc_w  =  calc_weights(discr_level, calc_c012s)
    preds_w =  calc_weights(discr_level, preds_c012s)
    delt_argmax = calculate_deltas_signed(np.argmax(preds_w[:], axis=1), 
                                          np.argmax(calc_w[:], axis=1), discr_level)      
    delt_argmax_rad = delt_argmax * k2PI / (discr_level - 1)

    meanrad = np.mean(delt_argmax) * k2PI / (discr_level - 1)
    stdrad  = np.std(delt_argmax) * k2PI / (discr_level - 1)
    meanraderr = stats.sem(delt_argmax) * k2PI / (discr_level - 1)

    # Preparing the plot
    bins = np.max(delt_argmax) - np.min(delt_argmax) + 1
    plt.hist(delt_argmax_rad, histtype='step', bins=bins, color='black')
    plt.xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')
    plt.gca()

    table_vals=[[r'Classification: $C_0, C_1, C_2$'],
                [" "],
                [r"mean = {:0.3f}$\pm$ {:1.3f} [rad]".format(meanrad, meanraderr)],
                ["std = {:1.3f} [rad]".format(stdrad)]
                ]

    table = plt.table(cellText=table_vals,
                    colWidths = [0.40],
                    cellLoc="left",
                    loc='upper right')
    table.set_fontsize(14)

    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)

    plt.tight_layout()

    # Saving the plot
    plt.savefig(f"{output_path}.pdf")
    plt.savefig(f"{output_path}.png")
    plt.savefig(f"{output_path}.eps")
    print(f"The plot has been saved as {output_path}")

    plt.clf()