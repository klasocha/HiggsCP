from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork, regr_argmaxs_loss
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from utilities.cpmix_utils import weight_fun


def draw_distribution(x, y, title, output_path, filename, color=None, 
                      info_table=None, multiple=False):
    if not multiple:
        plt.plot(x, y, color=color)
    else:
        for i in range(10):
            plt.plot(x, y[i])
    plt.title(title)
    plt.xlabel(r"${\alpha^{CP}}$ [idx]", loc="right")
    if multiple:
        plt.ylabel("Wt", rotation=0, labelpad=20)
    else:
        plt.ylabel(r"$\sum_{i=0}^N Wt_i$", rotation=0, labelpad=20)
    if info_table is not None:
        table_vals=[[f"Hypothesis class: {info_table[0]}"],
                [f"Argmax class: {info_table[1]}"]]
        table = plt.table(cellText=table_vals, colWidths = [0.40],
                          cellLoc="left", loc='upper right')
        table.set_fontsize(10)
        for _, cell in table.get_celld().items():
            cell.set_linewidth(0)
    plt.tight_layout()
    for format in ["pdf", "png", "eps"]:
        plt.savefig(os.path.join(output_path, f"{filename}.{format}"))
    print(f"The plot has been saved as {output_path}")
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
    unwt_path = os.path.join(args.IN, f"unwt_multiclass_{args.NUM_CLASSES}.npy") 
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
        preds =  calc_weights(discr_level, c012s)
    
    if args.TRAINING_METHOD == "regr_c012s":
        preds = calc_weights(discr_level, preds)

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Recomputing hypothesis index if the level of discretisation is different
    # from the number of classes the model works with
    if args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"]:
        hypothesis = round(hypothesis / n_classes * discr_level)

    # Creating a plot showing the summed distribution of Wt
    draw_distribution(
        x=np.linspace(0, discr_level - 1, num=discr_level),
        y=np.sum(preds, axis=0),
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_hyp_{hypothesis}_summed_dist",
        title="Summed distribution",
        color="black",
        info_table=[hypothesis, np.argmax(np.sum(preds, axis=0))])

    # Creating a plot showing some sample events predictied by the model
    draw_distribution(
        x=np.linspace(0, discr_level - 1, num=discr_level),
        y=preds,
        output_path=args.OUT,
        filename=f"{args.TRAINING_METHOD}_hyp_{hypothesis}_samples",
        title="Event spin weight",
        multiple=True)