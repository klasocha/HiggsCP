from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork, regr_argmaxs_loss
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utilities.cpmix_utils import weight_fun
import matplotlib.ticker as ticker 


def draw_distribution(x, y, title, output_path, filename, true_weights=None, color=None, 
                      info_table=None, multiple=False):
    
    if not multiple:
        fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 3])
        fig.set_size_inches(9, 6)
        ax2.plot(np.arange(len(x)), y, color=color[0], label="Predicted")
        ax2.plot(np.arange(len(x)), true_weights, linestyle="dotted", color=color[1], label="True")
        ax2.legend()
        ax2.set_ylabel(r"$\sum_{i=0}^N Wt_i$", rotation=0, labelpad=20)
        ax2.set_title(title)

        table_vals=[[f"Hypothesis idx: {info_table[0]}" + \
                    " (" + r"${{\alpha^{CP}}_{max}}$" + f" = {info_table[2]:0,.2f} rad)"],
                    [f"Predicted idx: {info_table[1]}" + \
                    " (" + r"${{\alpha^{CP}}_{max}}$" + f" = {info_table[3]:0,.2f} rad)"],
                    [f"Relative amplitude: {info_table[4]:0,.2f}"],
                    [r"${{\chi^2}/Nf}$" + f" = {info_table[5]:0,.2f}"]]
        
        ax1.axis('off')
        ax1.axis('tight')
        table = ax1.table(cellText=table_vals, colWidths = [0.6],
                          cellLoc="left", loc='upper left')
        table.set_fontsize(12)
        
        for _, cell in table.get_celld().items():
            cell.set_linewidth(0)
    else:
        fig, ax2 = plt.subplots(1)
        fig.set_size_inches(9, 6)
        for i in range(5):
            ax2.plot(np.arange(len(x)), y[np.random.randint(len(y))])
        ax2.set_ylabel("Wt", rotation=0, labelpad=20)
        ax2.set_title(title)

    ax2.set_xlabel(r"${\alpha^{CP}}$ [idx]", loc="right")    
    plt.xticks(np.arange(len(x)), x)
    if len(x) > 31:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(int(len(x) / 15), 1))

    plt.tight_layout()
    for format in ["pdf", "png", "eps"]:
        plt.savefig(os.path.join(os.path.normpath(output_path), f"{filename}.{format}"))
    print(f"The plot has been saved as {os.path.join(os.path.normpath(output_path), filename)}")
    plt.clf()


def calc_weights(num_classes, coeffs):
    k2PI = 2 * np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights


def test_on_unwt_events(args):
    """ Feed a pretrained NN with unweighted events (the whole data set is used)
    filtered according to a chosen hypothesis and create a double check plot 
    showing the summed distribution of the predicted weights """

    discr_level = int(args.NBINS) if args.NBINS is not None and \
        args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"] else int(args.NUM_CLASSES)
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
    unwt_path = os.path.join(os.path.normpath(args.IN), f"unwt_multiclass_{args.NUM_CLASSES}.npy") 
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

    # Loading weights and making predictions
    if args.TRAINING_METHOD in ["soft_weights", "regr_weights", "regr_c012s"]:
        model.load_weights(os.path.join(
            "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
            "model_state", "model.weights.h5"))
        preds = model.predict(X)

    if args.TRAINING_METHOD == "soft_c012s":
        c012s = np.zeros((X.shape[0], 3))
        for i in range(3):
            model.load_weights(os.path.join(
            "results", args.TRAINING_METHOD, 
            os.path.normpath(f"{args.MODEL_LOCATION}_c{i}"), 
            "model_state", "model.weights.h5"))
            coefficients = model.predict(X)
            c012s[:, i] = np.argmax(coefficients, axis=1)
            if i == 0:
                c012s[:, i] = c012s[:, i] * (2. / n_classes)
            else:
                c012s[:, i] = c012s[:, i] * (2. / n_classes) - 1.0
        preds = calc_weights(discr_level, c012s)
    
    if args.TRAINING_METHOD == "regr_c012s":
        preds = calc_weights(discr_level, preds)

    # Loading the true coefficients and calculating true weights
    true_c012s = read_np("data/c012s.npy")
    true_c012s = true_c012s[unwt == 1.0]
    true_weights = calc_weights(discr_level, true_c012s)

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Recomputing hypothesis index if the level of discretisation is different
    # from the number of classes the model works with
    if args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"]:
        hypothesis = round(hypothesis / n_classes * discr_level)

    # Normalising weights to the probability distribution
    preds = preds / np.sum(preds, axis=1).reshape((preds.shape[0], 1))
    true_weights = true_weights / np.sum(true_weights, axis=1).reshape((true_weights.shape[0], 1))

    # Creating a plot showing the summed distribution of Wt and computing the needed values
    predicted_argmax = np.argmax(np.sum(preds, axis=0)) 
    summed_wt = np.sum(preds, axis=0)
    summed_true_wt = np.sum(true_weights, axis=0)
    min_summed_wt, max_summed_wt = np.min(summed_wt), np.max(summed_wt)
    relative_amplitude = 2 * (max_summed_wt - min_summed_wt) / (max_summed_wt + min_summed_wt)
    chi2_nf = np.sum(np.square(summed_true_wt - summed_wt)) / discr_level 

    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(summed_wt[:-1], int((discr_level - 1) / 2)),
        true_weights=np.roll(summed_true_wt[:-1], int((discr_level - 1) / 2)),
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_hyp_{hypothesis}_summed_dist",
        title="Summed distribution",
        color=["black", "red"],
        info_table=[hypothesis, 
                    predicted_argmax,
                    hypothesis / (discr_level - 1) * 2 * np.pi, 
                    predicted_argmax / (discr_level - 1) * 2 * np.pi,
                    relative_amplitude,
                    chi2_nf])

    # Creating a plot showing some sample events predictied by the model
    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(preds[:, :-1], axis=1, shift=int((discr_level - 1) / 2)),
        output_path=args.OUT,
        filename=f"{args.TRAINING_METHOD}_hyp_{hypothesis}_samples",
        title="Event spin weight",
        multiple=True)