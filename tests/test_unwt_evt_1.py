# Testing model predictions on the events filtered by specific hypotheses
# and an unweighted events hits mask (soft/regr_weights, soft/regr_c012s)

from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork
import matplotlib.pyplot as plt
from utilities.cpmix_utils import weight_fun
import matplotlib.ticker as ticker 


def draw_distribution(x, y, title, output_path, filename, true_weights=None, color=None, 
                      info_table=None, multiple=False):
    
    if not multiple:
        fig, (ax1, ax2) = plt.subplots(2, height_ratios=[1, 4])
        fig.set_size_inches(10, 6)
        
        for i in range(len(y)):
            ax2.plot(np.arange(len(x)), y[i], color=color[i % len(color)], 
                     label="Predicted (" + r"${{\alpha^{CP}_{max}}}$" + f"={round(info_table[3][i], 1)})")
            ax2.plot(np.arange(len(x)), true_weights[i], color=color[i % len(color)], linestyle="dotted", 
                     label="True (" + r"${{\alpha^{CP}_{max}}}$" + f"={round(info_table[2][i], 1)})")
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylabel(r"$\sum_{i=0}^N Wt_i$", rotation=0, labelpad=20)
        ax2.set_title(title)

        for i in range(2, 4):
            info_table[i] = [round(value, 1) for value in info_table[i]]
        
        info_table[4] = [round(value, 2) for value in info_table[4]]

        table_vals=[["Hypothesis idx: " + ", ".join(f"{num}" for num in info_table[0]) + \
                    " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                        ", ".join(f"{num}" for num in info_table[2]) + " rad)"],

                    ["Predicted idx: " + ", ".join(f"{num}" for num in info_table[1]) + \
                    " (" + r"${{\alpha^{CP}}_{max}}$" + " = " + \
                        ", ".join(f"{num}" for num in info_table[3]) + " rad)"],
                    
                    ["Relative amplitude: " + ", ".join(f"{num}" for num in info_table[4])]]
        
        ax1.axis('off')
        ax1.axis('tight')
        table = ax1.table(cellText=table_vals, colWidths = [1.0],
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
    filtered according to chosen hypotheses and create a double check plot 
    showing the summed distribution of the predicted weights """

    discr_level = int(args.NBINS) if args.NBINS is not None and \
        args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"] else int(args.NUM_CLASSES)
    n_classes = int(args.NUM_CLASSES)

    # Parsing chosen hypotheses (e.g. --hypothesis "0-5-23")
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

    # Filtering the features according to the chosen hypotheses
    # defining the unweighted events mask
    features = []
    for hyp in hypotheses:
        mask = unwt[:, hyp]
        features.append(X[mask == 1.0])

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
    preds = []

    if args.TRAINING_METHOD in ["soft_weights", "regr_weights", "regr_c012s"]:
        model.load_weights(os.path.join(
            "results", args.TRAINING_METHOD, args.MODEL_LOCATION, 
            "model_state", "model.weights.h5"))
        for i in range(len(hypotheses)):
            preds.append(model.predict(features[i]))
    
    if args.TRAINING_METHOD == "soft_c012s":
        for i in range(len(hypotheses)):
            c012s = np.zeros((features[i].shape[0], 3))
            for j in range(3):
                model.load_weights(os.path.join(
                "results", args.TRAINING_METHOD, 
                os.path.normpath(f"{args.MODEL_LOCATION}_c{j}"), 
                "model_state", "model.weights.h5"))
                coefficients = model.predict(features[i])
                c012s[:, j] = np.argmax(coefficients, axis=1)
                if j == 0:
                    c012s[:, j] = c012s[:, j] * (2. / (n_classes - 1))
                else:
                    c012s[:, j] = c012s[:, j] * (2. / (n_classes - 1)) - 1.0
            preds.append(calc_weights(discr_level, c012s))
    
    if args.TRAINING_METHOD == "regr_c012s":
        for i in range(len(hypotheses)):
            preds[i] = calc_weights(discr_level, preds[i])

    # Loading the true coefficients and calculating true weights
    true_c012s = read_np(os.path.join(args.IN, "c012s.npy"))
    true_weights = []
    for hyp in hypotheses:
        c012s = true_c012s[unwt[:, hyp] == 1.0]
        true_weights.append(calc_weights(discr_level, c012s))

    # Creating a directory for storing the plots
    if not os.path.exists(os.path.normpath(args.OUT)):
        os.makedirs(os.path.normpath(args.OUT))

    # Recomputing hypothesis index if the level of discretisation is different
    # from the number of classes the model works with
    if args.TRAINING_METHOD in ["soft_c012s", "regr_c012s"]:
        for i in range(len(hypotheses)):
            hypotheses[i] = round(hypotheses[i] / (n_classes - 1) * (discr_level - 1))

    # Removing predictions containing negative weights
    preds_without_neg = []
    negs_n = []

    for i in range(len(hypotheses)):
        print(f"Hypothesis #{hypotheses[i]}")
        negs = np.where(preds[i] < 0, True, False)
        negs = np.sum(negs, axis=1)
        negs = np.where(negs > 0, True, False)
        negs_n.append(np.sum(negs))
        if negs_n[i] > 0:
            preds_without_neg.append(preds[i][negs == False])
        print(f"{negs_n[i]} (out of {len(preds[i])}) predictions lead to negative weights")

        # Normalising weights to the probability distribution
        if negs_n[i] == 0:
            print("Normalisation will be applied on predictions")
            preds[i] = preds[i] / np.sum(preds[i], axis=1).reshape((preds[i].shape[0], 1))
        else:
            print("Softmax normalisation will be applied on predictions as some",
                "of them contain negative weights")
            preds[i] = np.exp(preds[i]) / np.sum(
                np.exp(preds[i]), axis=1).reshape((preds[i].shape[0], 1))
            if np.sum(np.sum(preds_without_neg[i], axis=1).reshape(
                (preds_without_neg[i].shape[0], 1)) == 0) > 0:
                print("Softmax normalisation will be applied on predictions without",
                "negative weights as some of them sum up to zero")
                preds_without_neg[i] = np.exp(preds_without_neg[i]) / np.sum(
                    np.exp(preds_without_neg[i]), axis=1).reshape(
                        (preds_without_neg[i].shape[0], 1))
            else:
                print("Normalisation will be applied on predictions without negative weights")
                preds_without_neg[i] = preds_without_neg[i] / np.sum(
                    preds_without_neg[i], axis=1).reshape(
                        (preds_without_neg[i].shape[0], 1))
            
        true_weights[i] = true_weights[i] / np.sum(
            true_weights[i], axis=1).reshape((true_weights[i].shape[0], 1))
        print()

    # Computing the needed values
    summed_wt, predicted_argmax, summed_true_wt, relative_amplitude = \
        [], [], [], []

    for i in range(len(hypotheses)):
        summed_wt.append(np.sum(preds[i], axis=0))
        predicted_argmax.append(np.argmax(summed_wt[i]))
        summed_true_wt.append(np.sum(true_weights[i], axis=0))

        # Normalising summed distributions (to be able to compare them on the same plot)
        summed_wt[i] = summed_wt[i] / np.sum(summed_wt[i])
        summed_true_wt[i] = summed_true_wt[i] / np.sum(summed_true_wt[i])

        min_summed_wt, max_summed_wt = np.min(summed_wt[i]), np.max(summed_wt[i])
        relative_amplitude.append(2 * (max_summed_wt - min_summed_wt) / (max_summed_wt + min_summed_wt))
    
    summed_wt, summed_true_wt = np.array(summed_wt), np.array(summed_true_wt)

    # Creating a plot showing the summed distribution of Wt
    draw_distribution(
        x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
        y=np.roll(summed_wt[:, :-1], int((discr_level - 1) / 2), axis=1),
        true_weights=np.roll(summed_true_wt[:, :-1], int((discr_level - 1) / 2), axis=1),
        output_path=args.OUT,
        filename=f"{args.TRAINING_METHOD}_hyp_{args.HYPOTHESIS.replace('-', '_')}_summed_dist",
        title="Summed distribution",
        color=["black", "red", "blue"],
        info_table=[hypotheses, 
                    predicted_argmax,
                    np.array(hypotheses) / (discr_level - 1) * 2 * np.pi, 
                    np.array(predicted_argmax) / (discr_level - 1) * 2 * np.pi,
                    relative_amplitude])

    # Creating a plot showing some sample events predictied by the model
    for i in range(len(hypotheses)):
        draw_distribution(
            x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
            y=np.roll(preds[i][:, :-1], axis=1, shift=int((discr_level - 1) / 2)),
            output_path=args.OUT,
            filename=f"{args.TRAINING_METHOD}_hyp_{hypotheses[i]}_samples",
            title="Event spin weight",
            multiple=True)
    
    # Plotting the same for preprocessed predictions (those containing
    # negative weights are set to zero)
    if np.sum(negs_n) > 0:
        summed_wt, predicted_argmax, relative_amplitude = [], [], []
        for i in range(len(hypotheses)):
            summed_wt.append(np.sum(preds_without_neg[i], axis=0))
            predicted_argmax.append(np.argmax(summed_wt[i]))
            
            # Normalising summed distributions (to be able to compare them on the same plot)
            summed_wt[i] = summed_wt[i] / np.sum(summed_wt[i])

            min_summed_wt, max_summed_wt = np.min(summed_wt[i]), np.max(summed_wt[i])
            relative_amplitude.append(2 * (max_summed_wt - min_summed_wt) / (max_summed_wt + min_summed_wt))

        summed_wt = np.array(summed_wt)
        
        draw_distribution(
            x=np.roll(np.arange(0, discr_level - 1), int((discr_level - 1) / 2)),
            y=np.roll(summed_wt[:, :-1], int((discr_level - 1) / 2), axis=1),
            true_weights=np.roll(summed_true_wt[:, :-1], int((discr_level - 1) / 2), axis=1),
            output_path=args.OUT,
            filename=f"{args.TRAINING_METHOD}_hyp_{args.HYPOTHESIS.replace('-', '_')}_summed_dist_without_neg",
            title="Summed distribution",
            color=["black", "red", "blue"],
            info_table=[hypotheses, 
                        predicted_argmax,
                        np.array(hypotheses) / (discr_level - 1) * 2 * np.pi, 
                        np.array(predicted_argmax) / (discr_level - 1) * 2 * np.pi,
                        relative_amplitude])