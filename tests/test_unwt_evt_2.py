# Testing model predictions on the events filtered by a specific hypothesis
# and an unweighted events hits mask (soft/regr_argmaxs)

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
    lambda_fun(data)    
    return hits


def draw_distribution(preds, x, true_values, title, output_path, filename, 
                      color=None, info_table=None):
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 4])
    fig.set_size_inches(10, 6)

    for i in range(len(preds)):
        ax2.step(np.arange(len(x)), preds[i], color=color[i % len(color)], where="mid", 
                 label="Predicted (" + r"${{\alpha^{CP}_{max}}}$" + f"={round(info_table[4][i], 1)})")
        ax2.step(np.arange(len(x)), true_values[i], color=color[i % len(color)], where="mid", 
                 label="True (" + r"${{\alpha^{CP}_{max}}}$" + f"={round(info_table[6][i], 1)})",
                 linestyle="dotted")
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylabel("Entries", rotation=0, labelpad=10, loc="top")
    ax2.set_title(title)

    info_table[2] = [round(value, 2) for value in info_table[2]]
    for i in [1, 4, 6]:
        info_table[i] = [round(value, 1) for value in info_table[i]]
    
    table_vals=[["Hypothesis idx: " + ", ".join(f"{num}" for num in info_table[0]) + \
                " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                    ", ".join(f"{num}" for num in info_table[1]) + " rad)"],

                ["Actual hypothesis idx: " + ", ".join(f"{num}" for num in info_table[5]) + \
                " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                    ", ".join(f"{num}" for num in info_table[6]) + " rad)"],

                ["Predicted idx: " + ", ".join(f"{num}" for num in info_table[3]) + \
                " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                    ", ".join(f"{num}" for num in info_table[4]) + " rad)"],
                
                ["Relative amplitude: " + ", ".join(f"{num}" for num in info_table[2])]]
    
    ax1.axis('off')
    ax1.axis('tight')
    table = ax1.table(cellText=table_vals, colWidths = [1.0],
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


def test_on_unwt_events(args):
    """ Feed a pretrained NN with unweighted events (the whole data set is used)
    filtered according to a chosen hypothesis and create a double check plot 
    showing the distribution of the predicted alphaCP argmaxs """

    discr_level = int(args.NBINS) if args.NBINS is not None and \
        args.TRAINING_METHOD != "soft_argmaxs" else int(args.NUM_CLASSES)
    n_classes = int(args.NUM_CLASSES)
    
    # Parsing 3 chosen hypotheses (e.g. --hypothesis "0-5-23")
    hypotheses = args.HYPOTHESIS.split('-')
    hypotheses = [int(hyp) for hyp in hypotheses]

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
    features = []
    for hyp in hypotheses:
        mask = unwt[:, hyp]
        features.append(X[mask == 1.0])

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
    preds = []

    model.load_weights(os.path.join(
    "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
    "model_state", "model.weights.h5"))
    for i in range(len(hypotheses)):
        preds.append(model.predict(features[i]))
  
        if args.TRAINING_METHOD == "soft_argmaxs":
            preds[i] = np.argmax(preds[i], axis=1)
            preds[i] = preds[i] * 2 * np.pi / (n_classes - 1)
        
        if args.TRAINING_METHOD == "regr_argmaxs":
            # Shifting the predictions to the range [0, 2pi]
            for j in range(len(preds[i])):
                while preds[i][j] > (2 * np.pi):
                    preds[i][j] -= 2 * np.pi
                while preds[i][j] < 0:
                    preds[i][j] += 2 * np.pi

    # Loading and filtering true alphaCPmax values
    t_argmaxs = read_np(os.path.join(args.IN, "argmaxs.npy"))
    true_argmaxs = []
    for hyp in hypotheses:
        true_argmaxs.append(t_argmaxs[unwt[:, hyp] == 1.0])
    
    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Recomputing hypothesis index if the level of discretisation is different
    # from the number of classes the model works with
    if args.TRAINING_METHOD == "regr_argmaxs":
        for i in range(len(hypotheses)):
            hypotheses[i] = round(hypotheses[i] / (n_classes - 1) * (discr_level - 1))

    # Computing the needed values
    classes = np.linspace(0, 2 + 2/(discr_level - 1), (discr_level + 1)) * np.pi
    periodicity = True if args.TRAINING_METHOD == "regr_argmaxs" else False
    true_counts, preds_counts, relative_amplitude = [], [], []
    predicted_hypothesis, actual_hypothesis = [], []

    for i in range(len(hypotheses)):
        true_counts.append(bins_fun(classes, true_argmaxs[i], discr_level, True))
        preds_counts.append(bins_fun(classes, preds[i], discr_level, periodicity))
    
        # Getting rid of the last class (360° = 0°, bins_fun() takes into account this)
        preds_counts[i] = preds_counts[i][:-1]
        true_counts[i] = true_counts[i][:-1]
    
        preds_max_bin = preds_counts[i].max()
        preds_min_bin = preds_counts[i].min()
        relative_amplitude.append(2 * (preds_max_bin - preds_min_bin) / (preds_max_bin + preds_min_bin))
        predicted_hypothesis.append(np.argmax(preds_counts[i]))
        actual_hypothesis.append(np.argmax(true_counts[i]))

    preds_counts, true_counts = np.array(preds_counts), np.array(true_counts)

    # Creating a plot showing the distribution of argmaxs
    draw_distribution(
        preds=np.roll(preds_counts, int((discr_level - 1) / 2), axis=1),
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        true_values=np.roll(true_counts, int((discr_level - 1) / 2), axis=1),
        output_path=args.OUT,
        filename=f"{args.TRAINING_METHOD}_hyp_{args.HYPOTHESIS.replace('-', '_')}_dist",
        title="Distribution",
        color=["black", "red", "blue"],
        info_table=[hypotheses, 
            np.array(hypotheses) / (n_classes - 1) * 2 * np.pi, 
            relative_amplitude,
            predicted_hypothesis,
            np.array(predicted_hypothesis) / (n_classes - 1) * 2 * np.pi,
            actual_hypothesis,
            np.array(actual_hypothesis) / (n_classes - 1) * 2 * np.pi])