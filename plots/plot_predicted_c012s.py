"""
This program prepares the distribution of true and predicted coefficients 
C0,C1,C2:

    $ python main.py --action "plot" --option "C012S-FOR-PREDICTED" 
    --input "results/soft_c012s/variant_all/51_classes_c" 
    --output "plots/figures" --use_filtered_data --features "Variant-All"
    --training_method "soft_c012s" --dataset "test" --num_classes 51
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utilities.data_utils import read_np


def draw_weights_to_compare(calc, preds, output_path, c, args):
    plt.figure(figsize=(8, 6))
    plt.hist(calc, histtype='step', bins=np.arange(0, int(args.NUM_CLASSES) + 1), 
             color='black', label="Generated", linestyle="dotted")
    plt.hist(preds, histtype='step', bins=np.arange(0, int(args.NUM_CLASSES) + 1),
             color='red', label=f"Classification ${{C_0}}, {{C_1}}, {{C_2}}$")

    plt.xlabel(f"$\mathregular{{C_{c}}}$: Class index [idx]")
    plt.legend(loc='upper right')
    plt.tight_layout()

    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{output_path}_c{c}.{format}")
    print(f"The plot has been saved as {output_path}_c{c}")
    plt.clf()


def draw(args):
    """ Draw the predicted values and the true values of C0/C1/C2 """

    # Preparing the output directory
    num_classes = int(args.NUM_CLASSES)
        
    output_path = os.path.join(os.path.normpath(args.OUT), "c012s_predicted",
                            args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    filename = f"soft_c012s_{args.FEAT}_nc_{num_classes}_{filtered}"
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
    preds_c0s = np.argmax(preds_hits_c0s, axis=1)
    calc_c0s = np.argmax(calc_hits_c0s, axis=1)
    
    preds_c1s = np.argmax(preds_hits_c1s, axis=1)
    calc_c1s = np.argmax(calc_hits_c1s, axis=1)    
    
    preds_c2s = np.argmax(preds_hits_c2s, axis=1)
    calc_c2s = np.argmax(calc_hits_c2s, axis=1)
   
    calc_c012s = [calc_c0s, calc_c1s, calc_c2s]
    preds_c012s = [preds_c0s, preds_c1s, preds_c2s]

    for i in range(3):
        draw_weights_to_compare(calc_c012s[i], preds_c012s[i], output_path, i, args)