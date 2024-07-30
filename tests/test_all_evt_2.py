# Testing model predictions on all events (soft/regr_argmaxs)

from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork
from utilities.tf_model_keras_v2 import NeuralNetwork as NeuralNetwork_keras_v2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 


def bins_fun(classes, data, num_classes, periodicity):
    # Similar to hits_fun from cpmix_utils.py but data is an array and
    # hits contain the counts of data points
    hits = np.zeros(num_classes)
    
    def count_bins(e):
        if e < ((classes[0] + classes[1]) / 2):
            hits[0] += 1.0
            if periodicity:
                hits[num_classes - 1] += 1.0
        for i in range(1, num_classes):
            if ((classes[i-1] + classes[i]) / 2) <= e < \
                ((classes[i] + classes[i+1]) / 2):
                hits[i] += 1.0
                if i == num_classes - 1 and periodicity:
                    hits[0] += 1.0

    lambda_fun = np.vectorize(lambda x: count_bins(x))
    data = lambda_fun(data)    
    return hits


def draw_distribution(preds, x, true_values, title, output_path, filename, 
                      color=None, info_table=None):
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 6])
    fig.set_size_inches(9, 6)
    ax2.step(np.arange(len(x)), preds, color=color[0], where="mid", label="Predicted")
    ax2.step(np.arange(len(x)), true_values, color=color[1], where="mid", label="True")
    ax2.legend()
    ax2.set_ylabel("Entries", rotation=0, labelpad=10, loc="top")
    ax2.set_title(title)

    table_vals=[[f"Relative amplitude: {info_table[0]:0,.2f}"]]
       
    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=table_vals, colWidths = [0.6],
                      cellLoc="left", loc='upper left')
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


def test_on_all_events(args):
    """ Feed a pretrained NN with all events (the whole data set is used)
    and create a double check plot showing the distribution of the predicted 
    alphaCP argmaxs """

    discr_level = int(args.NBINS) if args.NBINS is not None and \
        args.TRAINING_METHOD != "soft_argmaxs" else int(args.NUM_CLASSES)
    n_classes = int(args.NUM_CLASSES)

    # Loading and standardising the input data (features)
    X_path = os.path.join(args.IN, f"rhorho_event_{args.FEAT}.obj")
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    X = X.cols[:, :-1] 
    mean = X.mean(0)
    std = X.std(0)
    X = (X - mean) / std

    # Preparing the model
    if args.KERAS == "v2":
        model = NeuralNetwork_keras_v2(
            configuration=args.TRAINING_METHOD, 
            n_features=X.shape[-1], 
            n_classes=n_classes,
            n_layers=int(args.LAYERS), 
            n_units_per_layer=int(args.SIZE),
            input_noise_rate=0.0,
            dropout_rate=float(args.DROPOUT),
            opt=args.OPT)
    else:
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
    "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
    "model_state", "model.weights.h5"))
    preds = model.predict(X)
  
    if args.TRAINING_METHOD == "soft_argmaxs":
        preds = np.argmax(preds, axis=1)
        preds = preds * 2 * np.pi / (n_classes - 1)
    
    if args.TRAINING_METHOD == "regr_argmaxs":
        # Shifting the predictions to the range [0, 2pi]
        for i in range(len(preds)):
            while preds[i] > (2 * np.pi):
                preds[i] -= 2 * np.pi
            while preds[i] < 0:
                preds[i] += 2 * np.pi

    # Loading true alphaCPmax values
    true_argmaxs = read_np(os.path.join(args.IN, "argmaxs.npy"))
    
    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Creating a plot showing the distribution of argmaxs and computing the needed values
    classes = np.linspace(0, 2 + 2/(discr_level - 1), (discr_level + 1)) * np.pi
    true_counts = bins_fun(classes, true_argmaxs, discr_level, True)
    periodicity = True if args.TRAINING_METHOD == "regr_argmaxs" else False
    preds_counts = bins_fun(classes, preds, discr_level, periodicity)

    # Getting rid of the last class (360° = 0°, bins_fun() takes into account this)
    preds_counts = preds_counts[:-1]
    true_counts = true_counts[:-1]
    
    preds_max_bin = preds_counts.max()
    preds_min_bin = preds_counts.min()
    relative_amplitude = 2 * (preds_max_bin - preds_min_bin) / (preds_max_bin + preds_min_bin)

    draw_distribution(
        preds=np.roll(preds_counts, int((discr_level - 1) / 2)),
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        true_values=np.roll(true_counts, int((discr_level - 1) / 2)),
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_all_events_dist",
        title="Distribution",
        color=["black", "red"],
        info_table=[relative_amplitude])