from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork
import matplotlib.pyplot as plt


def draw_distribution(preds, true_argmaxs, title, bins, output_path, filename, 
                      color=None, info_table=None):
    
    # if not multiple:
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 3])
    fig.set_size_inches(9, 6)
    ax2.hist(preds, bins=bins, color=color[0], label="Predicted", alpha=0.8)
    ax2.hist(true_argmaxs, bins=bins, color=color[1], linestyle="dotted", 
             histtype="step", label="True", linewidth=2.0)
    ax2.legend()
    ax2.set_ylabel("Entries", rotation=0, labelpad=20)
    ax2.set_title(title)

    table_vals=[[f"Hypothesis idx: {info_table[0]}" + \
                " (" + r"${{\alpha^{CP}_{max}}}$" + f" = {info_table[1]:0,.2f} rad)"],
                [f"Relative amplitude: {info_table[2]:0,.2f}"]]    
    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=table_vals, colWidths = [0.6],
                      cellLoc="left", loc='upper left')
    table.set_fontsize(12)
    
    for _, cell in table.get_celld().items():
        cell.set_linewidth(0)

    ax2.set_xlabel(r"${{\alpha}^{CP}_{max}}$ [rad]", loc="right")    

    plt.tight_layout()
    for format in ["pdf", "png", "eps"]:
        plt.savefig(os.path.join(os.path.normpath(output_path), f"{filename}.{format}"))
    print(f"The plot has been saved as {os.path.join(os.path.normpath(output_path), filename)}")
    plt.clf()


def test_on_unwt_events(args):
    """ Feed a pretrained NN with unweighted events (the whole data set is used)
    filtered according to a chosen hypothesis and create a double check plot 
    showing the distribution of the predicted weights """

    discr_level = int(args.NBINS) if args.NBINS is not None else int(args.NUM_CLASSES)
    n_classes = int(args.NUM_CLASSES)
    hypothesis = int(args.HYPOTHESIS)

    # Loading and standardising the input data (features)
    X_path = os.path.join(args.IN, f"rhorho_event_{args.FEAT}.obj")
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    X = X.cols[:, :-1] 
    mean = X.mean(0)
    std = X.std(0)
    X = (X - mean) / std

    # Loading the unweighted events weights
    unwt_path = os.path.join(os.path.normpath(args.IN), 
                             f"unwt_multiclass_{args.NUM_CLASSES}.npy") 
    unwt = read_np(unwt_path)

    # Filtering the features according to the chosen hypothesis
    # defining the unweighted events mask
    unwt = unwt[:, hypothesis]
    X = X[unwt == 1.0]

    # Preparing the model
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
    if args.TRAINING_METHOD == "regr_argmaxs":
        model.load_weights(os.path.join(
            "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
            "model_state", "model.weights.h5"))
        preds = model.predict(X)

    # Shifting the predictions to the range [0, 2pi]
    for i in range(len(preds)):
        while preds[i] > (2 * np.pi):
            preds[i] -= 2 * np.pi
        while preds[i] < 0:
            preds[i] += 2 * np.pi

    # Loading and filtering true alphaCPmax values
    true_argmaxs = read_np("data/argmaxs.npy")
    true_argmaxs = true_argmaxs[unwt == 1.0]

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Creating a plot showing the distribution of argmaxs and computing the needed values
    min_preds, max_preds = np.min(preds), np.max(preds)
    relative_amplitude = 2 * (max_preds - min_preds) / (max_preds + min_preds)

    draw_distribution(
        preds=preds,
        true_argmaxs=true_argmaxs,
        bins=discr_level,
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_hyp_{hypothesis}_summed_dist",
        title="Distribution",
        color=["black", "red"],
        info_table=[hypothesis, hypothesis / (n_classes - 1) * 2 * np.pi, 
            relative_amplitude])