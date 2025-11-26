# Testing model predictions on the unweighted events

from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork
import matplotlib.pyplot as plt
from utilities.cpmix_utils import weight_fun
import matplotlib.ticker as ticker 


def draw_distribution(x, y, title, output_path, filename, true_weights=None, color=None, 
                      info_table=None, data_format="v3"):
    
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 4])
    fig.set_size_inches(10, 6)

    ax2.plot(np.arange(len(x)), y, color=color, 
             label="Predicted (" + r"${{\alpha^{CP}_{max}}}$" + f"={round(info_table[1], 1)})")

    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylabel(r"$\sum_{i=0}^N Wt_i$", rotation=0, labelpad=20)
    ax2.set_title(title)

    info_table[2] = round(info_table[2], 2)

    table_vals=[
        ["Predicted idx: " + str(info_table[0]) + \
            " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                str(info_table[1]) + " rad)"],
        ["Relative amplitude: " + str(info_table[2])]
    ]
    
    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=table_vals, colWidths = [1.0], cellLoc="left", loc='upper left')
    table.set_fontsize(12)
    
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)

    ax2.set_xlabel(r"${\alpha^{CP}}$ [idx]", loc="right")    
    plt.xticks(np.arange(len(x)), x)
    if len(x) > 31:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(int(len(x) / 15), 1))

    plt.tight_layout()
    for format in ["pdf", "png", "eps"]:
        plt.savefig(os.path.join(os.path.normpath(output_path), f"{filename}.{format}"))
    print(f"The plot has been saved as {os.path.join(os.path.normpath(output_path), filename)}")
    plt.clf()


def test_on_unwt_events(args):
    """ Feed a pretrained NN with unweighted events (the whole data set is used)
    and create a double check plot showing the summed distribution of the predicted weights """

    discr_level = int(args.NUM_CLASSES)

    # Loading and standardising the input data (features)
    X_path = os.path.join(args.IN, f"rhorho_event_{args.FEAT}.obj")
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    X = X.cols[:, :-1] 
    mean = X.mean(0)
    std = X.std(0)
    X = (X - mean) / std

    # Preparing the model
    n_classes = int(args.NUM_CLASSES)
    model = NeuralNetwork(
        configuration=args.TRAINING_METHOD, 
        n_features=X.shape[-1], 
        n_classes=n_classes,
        n_layers=int(args.LAYERS), 
        n_units_per_layer=int(args.SIZE),
        input_noise_rate=0.0,
        dropout_rate=float(args.DROPOUT),
        opt=args.OPT)
    model.build()

    # Loading model weights and making predictions
    model.load_weights(os.path.join(
        "results", args.TRAINING_METHOD, args.MODEL_LOCATION, "model_state", "model.weights.h5"))
    preds = model.predict(X)
    plt.hist(preds, bins="auto")
    plt.show()
    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Computing the needed values
    dist_weights = 1.0
    summed_wt = np.sum(preds * dist_weights, axis=0)
    predicted_argmax = np.argmax(summed_wt)

    # Normalising summed distributions (to be able to compare them on the same plot)
    summed_wt = summed_wt / np.sum(summed_wt)
    min_summed_wt, max_summed_wt = np.min(summed_wt), np.max(summed_wt)
    relative_amplitude = 2 * (max_summed_wt - min_summed_wt) / (max_summed_wt + min_summed_wt)
    summed_wt = np.array(summed_wt)

    # Creating a plot showing the summed distribution of Wt
    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(summed_wt[:-1], int((discr_level - 1) / 2), axis=0),
        output_path=args.OUT,
        filename=f"{args.TRAINING_METHOD}_summed_dist",
        title="Summed distribution",
        color="black",
        info_table=[predicted_argmax,
                    np.array(predicted_argmax) / (discr_level - 1) * 2 * np.pi,
                    relative_amplitude],
        data_format=args.DATA_FORMAT)
    