"""
This program prepares a plot for normalised to probability spin weight Wt_norm, 
predicted by "soft_weights" (orange diamonds) and true (blue dots), 
as a function of the class index for two example events. 

    $ python main.py --action "plot" --option "WEIGHTS-FOR-PREDICTED" 
    --input "results/soft_weights/51_classes_variant_all/predictions/" 
    --output "plots/figures" --use_filtered_data --features "Variant-All" 
    --training_method "soft_weights" --dataset "test" --num_classes 51
"""

import os, random
import numpy as np
import matplotlib.pyplot as plt
from utilities.data_utils import read_np


def draw_weights_to_compare(calc, preds, output_path, args):
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(15, 5)    
    
    # Left subplot
    ax1.plot(np.arange(0, args.NUM_CLASSES), 
                calc[0], label="Generated")   
    ax1.step(np.arange(0, args.NUM_CLASSES), preds[0], where="mid",
            label="Classification: wt", color="orange")
    ax1.set_xlabel("Class index")
    ax1.set_ylabel(r'$Wt^{norm}$')
    ax1.legend(loc='upper right')
    
    # Right subplot
    ax2.plot(np.arange(0, args.NUM_CLASSES), 
                calc[1], label="Generated")   
    ax2.step(np.arange(0, args.NUM_CLASSES), preds[1], where="mid",
            label="Classification: wt", color="orange")
    ax2.set_xlabel("Class index")
    ax2.set_ylabel(r'$Wt^{norm}$')
    ax2.legend(loc='upper right')
    
    plt.subplots_adjust(wspace=0.4, bottom=0.2)

    # Showing and saving the plot
    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{output_path}.{format}")
    print(f"The plot has been saved as {output_path}")
    plt.clf()


def draw(args):
    """ Draw the predicted values and the true values of the weight(alphaCP) function """

    # Loading calculated and true weights
    filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
    dataset = filtered + '_' + args.DATASET
    calc_w  = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_calc.npy"))
    preds_w  = read_np(os.path.join(os.path.normpath(args.IN), f"{dataset}_preds.npy"))

    # Chosing randomly one event
    events = [random.randint(0, len(calc_w) - 1),
              random.randint(0, len(calc_w) - 1)]

    # Preparing the output directory 
    num_classes = int(args.NUM_CLASSES)
    output_path = os.path.join(os.path.normpath(args.OUT), "wt_predicted", 
                               args.TRAINING_METHOD, args.DATASET)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = f"soft_wt_{args.FEAT}_nc_{num_classes}_" + \
        f"{filtered}_samples_{events[0]}_{events[1]}"

    output_path = os.path.join(output_path, filename)

    # Creating plots for two sample events
    draw_weights_to_compare(calc_w[events], preds_w[events],
                            output_path, args)