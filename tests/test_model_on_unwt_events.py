from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork, regr_argmaxs_loss
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def draw_distribution(x, y, title, output_path, filename, color=None, multiple=False):
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
    plt.tight_layout()
    for format in ["pdf", "png", "eps"]:
        plt.savefig(os.path.join(output_path, f"{filename}.{format}"))
    print(f"The plot has been saved as {output_path}")
    plt.clf()


def test_on_unwt_events(args):
    """ Feed a pretrained NN with unweighted events (the whole data set is used)
    filtered according to a chosen hypothesis and create a double check plot 
    showing the summed distribution of the predicted weights """

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

    # Loading a trained model and making predictions
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
    model.load_weights(os.path.join(
        "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
        "model_state", "model.weights.h5"))
    preds = model.predict(X)
    print("Summed preds argmax:", np.argmax(np.sum(preds, axis=0)))

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Creating a plot showing the summed distribution of Wt
    draw_distribution(
        x=np.linspace(0, n_classes - 1, num=n_classes),
        y=np.sum(preds, axis=0),
        output_path=args.OUT,
        filename= f"{hypothesis}_summed_dist",
        title="Summed distribution",
        color="black")

    # Creating a plot showing the amplified summed distribution of Wt relative to
    # the weights of the chosen hypothesis
    hyp_vec = preds[:, hypothesis]
    ampl_preds = preds * hyp_vec[:, np.newaxis]
    draw_distribution(
        x=np.linspace(0, n_classes - 1, num=n_classes),
        y=np.sum(ampl_preds, axis=0),
        output_path=args.OUT,
        filename=f"{hypothesis}_amplified_dist",
        title="Amplified distribution",
        color="black")

    # Creating a plot showing some sample events predictied by the model
    draw_distribution(
        x=np.linspace(0, n_classes - 1, num=n_classes),
        y=preds,
        output_path=args.OUT,
        filename=f"{hypothesis}_sample_predictions",
        title="Event spin weight",
        multiple=True)