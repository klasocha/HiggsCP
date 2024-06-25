# Testing model predictions on all events (soft/regr_weights, soft/regr_c012s)

from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork
import matplotlib.pyplot as plt
from utilities.cpmix_utils import weight_fun
import matplotlib.ticker as ticker 


def draw_distribution(x, y, title, output_path, filename, true_weights=None, 
                      color=None, info_table=None, argmax_dist=False):
    
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 6])
    fig.set_size_inches(9, 6)
    ax2.plot(np.arange(len(x)), y, color=color[0], label="Predicted")
    ax2.plot(np.arange(len(x)), true_weights, linestyle="dotted", 
                color=color[1], label="True")
    ax2.legend()
    if argmax_dist:
        ax2.set_ylabel("Wt", rotation=0, labelpad=20)
    else:
        ax2.set_ylabel(r"$\sum_{i=0}^N Wt_i$", rotation=0, labelpad=20)
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
    
    filename = os.path.join(os.path.normpath(output_path), filename)
    for format in ["pdf", "png", "eps"]:
        plt.savefig(f"{filename}.{format}")
    print(f"The plot has been saved as {filename}")
    plt.clf()


def calc_weights(num_classes, coeffs):
    k2PI = 2 * np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights


def test_on_all_events(args):
    """ Feed a pretrained NN with all events (the whole data set is used)
    and create a double check plot showing the summed distribution of the 
    predicted weights """

    discr_level = int(args.NBINS) if args.NBINS is not None and \
        args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"] else \
        int(args.NUM_CLASSES)
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
                c012s[:, i] = c012s[:, i] * (2. / (n_classes - 1))
            else:
                c012s[:, i] = c012s[:, i] * (2. / (n_classes - 1)) - 1.0
        preds = calc_weights(discr_level, c012s)
    
    if args.TRAINING_METHOD == "regr_c012s":
        preds = calc_weights(discr_level, preds)

    # Loading the true coefficients and calculating true weights
    true_c012s = read_np(os.path.join(args.IN, "c012s.npy"))
    true_weights = calc_weights(discr_level, true_c012s)

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Testing for negative weights
    negs = np.where(preds < 0, True, False)
    negs = np.sum(negs, axis=1)
    negs = np.where(negs > 0, True, False)
    negs_n = np.sum(negs)
    # Removing predictions containing negative weights
    if negs_n > 0:
        preds_without_neg = preds[negs == False] 
    print(f"{negs_n} (out of {len(preds)}) predictions lead to negative weights")

    # Normalising weights to the probability distribution
    if negs_n == 0:
        print("Normalisation will be applied on predictions")
        preds = preds / np.sum(preds, axis=1).reshape((preds.shape[0], 1))
    else:
        print("Softmax normalisation will be applied on predictions as some",
              "of them contain negative weights")
        preds = np.exp(preds) / np.sum(np.exp(preds), axis=1).reshape((preds.shape[0], 1))
        if np.sum(np.sum(preds_without_neg, axis=1).reshape((preds_without_neg.shape[0], 1)) == 0) > 0:
            print("Softmax normalisation will be applied on predictions without",
              "negative weights as some of them sum up to zero")
            preds_without_neg = np.exp(preds_without_neg) / np.sum(
                np.exp(preds_without_neg), axis=1).reshape((preds_without_neg.shape[0], 1))
        else:
            print("Normalisation will be applied on predictions without negative weights")
            preds_without_neg = preds_without_neg / np.sum(
                preds_without_neg, axis=1).reshape((preds_without_neg.shape[0], 1))
    
    true_weights = true_weights / np.sum(true_weights, axis=1).reshape(
        (true_weights.shape[0], 1))

    # Creating a plot showing the summed distribution of Wt and computing the needed values
    summed_wt = np.sum(preds, axis=0)
    summed_true_wt = np.sum(true_weights, axis=0)
    min_summed_wt, max_summed_wt = np.min(summed_wt), np.max(summed_wt)
    relative_amplitude = 2 * (max_summed_wt - min_summed_wt) / \
        (max_summed_wt + min_summed_wt)

    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(summed_wt[:-1], int((discr_level - 1) / 2)),
        true_weights=np.roll(summed_true_wt[:-1], int((discr_level - 1) / 2)),
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_all_events_summed_dist",
        title="Summed distribution",
        color=["black", "red"],
        info_table=[relative_amplitude])

    # Creating a plot showing Wt argmax distribution
    argmax_wt, _ = np.histogram(np.argmax(preds, axis=1), 
                                bins=np.arange(0, discr_level  + 1))
    argmax_true_wt, _ = np.histogram(np.argmax(true_weights, axis=1), 
                                    bins=np.arange(0, discr_level + 1))
    # np.argmax(preds) returns 1 value, so we need to take into account
    # periodicity of preds (360° = 0°)
    common_counts = argmax_wt[0] + argmax_wt[discr_level - 1] 
    argmax_wt[0], argmax_wt[discr_level - 1] = common_counts, common_counts
    common_counts = argmax_true_wt[0] + argmax_true_wt[discr_level - 1] 
    argmax_true_wt[0], argmax_true_wt[discr_level - 1] = common_counts, common_counts

    min_argmax_wt, max_argmax_wt = np.min(argmax_wt), np.max(argmax_wt)
    relative_amplitude = 2 * (max_argmax_wt - min_argmax_wt) / \
        (max_argmax_wt + min_argmax_wt)
    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(argmax_wt[:-1], int((discr_level - 1) / 2)),
        true_weights=np.roll(argmax_true_wt[:-1], int((discr_level - 1) / 2)),
        output_path=args.OUT,
        filename= f"{args.TRAINING_METHOD}_all_events_argmax_dist",
        title="Wt argmax distribution",
        color=["black", "red"],
        info_table=[relative_amplitude],
        argmax_dist=True)
    
    # Plotting the same for preprocessed predictions (those containing
    # negative weights are set to zero)
    if negs_n > 0:
        summed_wt = np.sum(preds_without_neg, axis=0)

        # Compensating the lack of predictions containing negative weights to
        # keep the OY axis scale relative to summed_true_wt
        summed_wt += (negs_n / discr_level) 
        
        min_summed_wt, max_summed_wt = np.min(summed_wt), np.max(summed_wt)
        relative_amplitude = 2 * (max_summed_wt - min_summed_wt) / \
            (max_summed_wt + min_summed_wt)

        draw_distribution(
            x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
            y=np.roll(summed_wt[:-1], int((discr_level - 1) / 2)),
            true_weights=np.roll(summed_true_wt[:-1], int((discr_level - 1) / 2)),
            output_path=args.OUT,
            filename= f"{args.TRAINING_METHOD}_all_events_summed_dist_without_neg",
            title="Summed distribution",
            color=["black", "red"],
            info_table=[relative_amplitude])

        argmax_wt, _ = np.histogram(np.argmax(preds_without_neg, axis=1),
                                    bins=np.arange(0, discr_level + 1))
        
        # Compensating the lack of predictions containing negative weights to
        # keep the OY axis scale relative to summed_true_wt
        argmax_wt += (round(negs_n / discr_level))

        # np.argmax(preds) returns 1 value, so we need to take into account
        # periodicity of preds (360° = 0°)
        common_counts = argmax_wt[0] + argmax_wt[discr_level - 1] 
        argmax_wt[0], argmax_wt[discr_level - 1] = common_counts, common_counts

        min_argmax_wt, max_argmax_wt = np.min(argmax_wt), np.max(argmax_wt)
        relative_amplitude = 2 * (max_argmax_wt - min_argmax_wt) / \
            (max_argmax_wt + min_argmax_wt)
        
        draw_distribution(
            x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
            y=np.roll(argmax_wt[:-1], int((discr_level - 1) / 2)),
            true_weights=np.roll(argmax_true_wt[:-1], int((discr_level - 1) / 2)),
            output_path=args.OUT,
            filename= f"{args.TRAINING_METHOD}_all_events_argmax_dist_without_neg",
            title="Wt argmax distribution",
            color=["black", "red"],
            info_table=[relative_amplitude],
            argmax_dist=True)