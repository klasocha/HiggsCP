import argparse, os
from utilities.download_original_data import download as download_original_data
from utilities.tf_model import run as train_model
from utilities.tf_model_keras_v2 import run as train_model_keras_v2
from utilities.prepare_data import prepare_data
from plots.plot_phistar_distribution import draw as phistar_dist 
from plots.plot_popts_rhorho import draw as c012s_weight
from plots.plot_calc_c012s import draw as c012s_dist
from plots.plot_weights_with_c012s import draw as weights_with_c012s
from plots.plot_unwt_weights import draw as unwt_weights
from plots.plot_predicted_wt import draw as predicted_weights
from plots.plot_predicted_c012s import draw as predicted_c012s
from plots.results_analysis_1 import draw as results_analysis_1
from plots.results_analysis_2 import draw as results_analysis_2
from plots.results_analysis_3 import draw as results_analysis_3
from plots.results_analysis_4 import draw as results_analysis_4
from plots.results_analysis_5 import draw as results_analysis_5
from tests.test_parsed_data import test_parsed_data, show_example_records
from tests.test_unwt_evt_1 import test_on_unwt_events as test_on_unwt_events_1
from tests.test_sum_dist_1 import test_on_unwt_events as test_on_already_unwt_events_1
from tests.test_unwt_evt_2 import test_on_unwt_events as test_on_unwt_events_2
from tests.test_all_evt_1 import test_on_all_events as test_on_all_events_1
from tests.test_all_evt_2 import test_on_all_events as test_on_all_events_2
from tests.test_labels import test_labels
from utilities.prepare_rhorho import prepare_rhorho
from utilities.prepare_z import prepare_z


# ====================== HANDLING COMMAND LINE ARGUMENTS =======================
# Initialising a parser handling all the commaind-line arguments and options
parser = argparse.ArgumentParser(
  prog='Higgs Boson CP Classifier',
  description='Download data and train the classifier for the Higgs Boson' + 
    'CP problem')

# Arguments used by src_py/download_data_rhorho.py
parser.add_argument("-i", "--input", dest="IN", type=os.fspath, 
                    help="data path", default="temp_data")
parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", 
                    action="store_true", default=False, 
                    help="overwriting existing data")

# Arguments used by src_py/cpmix_utils.py
parser.add_argument("--num_classes", dest="NUM_CLASSES", type=int, default=0,
                    help="number of classes used for discretisation")
parser.add_argument("--reuse_weights", dest="REUSE_WEIGHTS", 
                    action="store_true", default=True,
                    help="set this flag to True if you want to reuse the " +
                    "calculated weights")
parser.add_argument("--hits_c012s", dest="HITS_C012s", 
                    choices=["hits_c0s", "hits_c1s",  "hits_c2s"], 
                    default="hits_c0s", help="which coefficients (C0, C1 or C2)"
                    + " to choose as labels")

# Arguments used by src_py/data_utils.py
parser.add_argument("--miniset", dest="MINISET", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], 
                    default=False, help="using the small version of the " + 
                    "training data set")

# Arguments used by src_py/rhorho.py
parser.add_argument("--beta",  type=float, dest="BETA", 
                    help="the beta parameter value for polynomial smearing", 
                    default=0.0)
parser.add_argument("-f", "--features", dest="FEAT", help="Features", 
                    default="Variant-All")
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], 
                    default="A")

# Arguments used by src_py/tf_model.py
parser.add_argument("--training_method", dest="TRAINING_METHOD", 
                    choices=["soft_weights", "soft_c012s",  "soft_argmaxs", 
                             "regr_c012s", "regr_weights", "regr_argmaxs"], 
                    default="soft_weights", 
                    help="training method (the loss function type)")
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, 
                    help="number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", 
                    default=100)
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.0,
                    help="dropout probability " +
                    "(applied during the training process)")
parser.add_argument("-o", "--optimizer", dest="OPT", 
                    choices=["GradientDescentOptimizer", "AdadeltaOptimizer", 
                             "AdagradOptimizer", "ProximalAdagradOptimizer", 
                             "AdamOptimizer", "FtrlOptimizer", 
                             "ProximalGradientDescentOptimizer", 
                             "RMSPropOptimizer"], 
                    default="AdamOptimizer", help="TensorFlow optimiser")
parser.add_argument("--learning_rate", dest="LEARNING_RATE", type=float, default=0.001,
                    help="learning rate for the optimiser")
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3,
                    help="the number of epochs used during the training process")
parser.add_argument("--delt_classes", dest="DELT_CLASSES", type=int, default=0, 
                    help=("maximum allowed difference between the predicted " + 
                          "class and the true class for an event to be " + 
                          "considered correctly classified."))

# Arguments downloading the original data
parser.add_argument("--download_original_data", dest="DOWNLOAD_ORIGINAL", 
                    help="downloading the original data",
                    action="store_true", default=False)

parser.add_argument("--use_unweighted_events", dest="USE_UNWEIGHTED_EVENTS", 
                    action="store_true", help="applying the unweighted events" +
                    " for training (Monte Carlo)", default=False)

# Keras & TFv2 arguments
parser.add_argument("--model_location", dest="MODEL_LOCATION", 
                    help="name of the directory in \"results/\" containing the " 
                    + "model state (weights, metadata)")
parser.add_argument("--use_filtered_data", dest="USE_FILTERED_DATA", 
                    help="picking only those vectors having \"pt\" value " +
                    "greater than 20", action="store_true", default=False)

# Plot arguments
plot_types = {
    # Variant-1.1 should be prepared in advance
    "PHISTAR-DISTRIBUTION" : phistar_dist, 
    "C012S-WEIGHT" : c012s_weight,
    "C012S-DISTRIBUTION" : c012s_dist,
    "WEIGHTS-FOR-EVENT-VIA-C012": weights_with_c012s,
    "UNWEIGHTED-EVENTS-WEIGHTS": unwt_weights,
    # "soft_weights", "regr_weights"
    "RESULTS_ANALYSIS_1": results_analysis_1, 
    # "soft_c012s"
    "RESULTS_ANALYSIS_2": results_analysis_2,
    # "soft_argmaxs" 
    "RESULTS_ANALYSIS_3": results_analysis_3, 
    # "regr_c012s"
    "RESULTS_ANALYSIS_4": results_analysis_4, 
    # "regr_argmaxs"
    "RESULTS_ANALYSIS_5": results_analysis_5, 
    # predicted weights ("soft_weights") vs true weights
    "WEIGHTS-FOR-PREDICTED": predicted_weights,
    # predicted c012s ("soft_c012s") vs true c012s 
    "C012S-FOR-PREDICTED" : predicted_c012s 
}
parser.add_argument("--output", dest="OUT", help="output path for plots", 
                    default="figures")
parser.add_argument("--format", dest="FORMAT", 
                    help='the format of the output plots ("png"/"pdf"/"eps")', 
                    default="png")
parser.add_argument("--show", dest="SHOW", action="store_true", 
                    help='use it to display the plots before saving them', 
                    default=False)
parser.add_argument("--option", dest="OPTION", choices=plot_types.keys(), 
                    default="PHISTAR-DISTRIBUTION",
                    help="specify what script for drawing the plots you " + 
                    "want to run")
parser.add_argument("--hypothesis", dest="HYPOTHESIS", default="0-4-46", 
                    help="Hypothesis: the alphaCP class (e.g. 02) or several " +
                    "classes \"#-#-#\"")
parser.add_argument("--without_unweighting", dest="WITHOUT_UNWEIGHTING",
                    action="store_true", default=False, 
                    help="do not unweight the events, use them as they are")
parser.add_argument("--dataset", dest="DATASET", 
                    help="dataset (train/valid/test)")
parser.add_argument("--binning", dest="NBINS", 
                    help="number of classes used for plotting histograms, " + 
                    "computing weights via C0/C1/C2, etc.")

# Test arguments
parser.add_argument("--source-1", dest="SOURCE_1",
                    help="the first directory containing data to be compared")
parser.add_argument("--source-2", dest="SOURCE_2", 
                    help="the second directory containing data to be compared")
parser.add_argument("--datasets", dest="DATASETS", default=2, type=int, 
                    help="number of datasets to prepare")

# Z-background experiment
parser.add_argument("--exp", dest="EXP", default="RhoRho", 
                    choices=["RhoRho", "Z"], 
                    help="Z for using Z-background data")

# New data format
parser.add_argument("--data_format", dest="DATA_FORMAT", default="v4",
                    choices=["v1", "v2", "v3", "v4"], 
                    help="input data format version (v1, v2 etc.)")

# Main controller
parser.add_argument("--action", dest="ACTION", 
                    choices=["download_and_prepare_original", 
                    "download_prepared_and_preprocess", "preprocess", "train", 
                    "continue_training", "predict_train_and_valid", "plot", 
                    "test_parsed_data", "test_model_on_unwt_events", 
                    "predict_test", "test_labels", "experimental",
                    "test_model_on_all_events"], 
                    default="train")
parser.add_argument("--keras", dest="KERAS", choices=["v2", "v3"], default="v3", 
                    help="the version of the Keras engine")

# Parsing the command-line arguments 
args = parser.parse_args()

# ======================== CONTROLING THE ML FLOW  =============================
if args.ACTION == "download_and_prepare_original":
    # $ python main.py --action "download_and_prepare_original" --input "data"
    download_original_data(args)
    if args.EXP == "Z":
        prepare_z(args)
    else:
        prepare_rhorho(args)

if args.ACTION == "download_prepared_and_preprocess":
    # $ python main.py --action "download_prepared_and_preprocess" --input 
    # "data" --features Variant-All --num_classes "51"
    prepare_data(args)

if args.ACTION == "preprocess":
    # $ python main.py --action "preprocess" --input "data" --features 
    # Variant-All --num_classes "51"
    prepare_data(args, preprocess_only=True)

if args.ACTION in ["train", "continue_training", "predict_train_and_valid", 
                   "predict_test"]:
    # 1. python main.py --action "train" --input "data" --num_classes "51" 
    # --epochs "2" --training_method "soft_weights" --model_location "model_1"
    
    # 2. python main.py --action "continue_training" --input "data" 
    # --num_classes "51" --epochs "3" --training_method "soft_weights" 
    # --model_location "model_1"
    
    # 3. python main.py --action "predict_train_and_valid" --input "data" 
    # --num_classes "51" --model_location "model_1"
    
    # 4. python main.py --action "predict_test" --input "data" --num_classes 
    # "51" --model_location "model_1"
    if args.KERAS == "v3":
        train_model(args)
    elif args.KERAS == "v2":
        train_model_keras_v2(args)
        
if args.ACTION == "plot":
    # Instructions are in the modules located in plots/
    plot_types[args.OPTION](args)

if args.ACTION == "test_parsed_data":
    # $ python main.py --action "test_parsed_data" --source-1 "data" --source-2
    # "data_original" --input "data_original"
    print(""" 
    This part was created to test 
        1. "prepare_utils.py", 
        2. "prepare_rhorho.py", 
        3. "download_data_rhorho.py".
    """)
    test_parsed_data(args)
    show_example_records(args)

if args.ACTION == "test_model_on_unwt_events":
    # $ python main.py --action "test_model_on_unwt_events" --input "data" 
    # --output "plots/figures/test_model_on_unwt_events" --num_classes "51" 
    # --hypothesis "0-10-15" --training_method "soft_weights" --model_location 
    # "model_1" --features "Variant-All"
    if args.TRAINING_METHOD not in ["regr_argmaxs", "soft_argmaxs"]:
        print(""" 
        This part was created to test the trained model by feeding it with
        the unweighted events and then creating a plot showing the summed
        distribution of the predicted weights.
        """)
        test_on_unwt_events_1(args)
        # test_on_already_unwt_events_1(args)
    else:
        print(""" 
        This part was created to test the trained model by feeding it with
        the unweighted events and then creating a plot showing the
        distribution of the predicted alphaCP max.
        """)
        test_on_unwt_events_2(args)

if args.ACTION == "test_model_on_all_events":
    # $ `python main.py --action "test_model_on_all_events" --input "data" 
    # --output "plots/figures/test_model_on_all_events" --num_classes "51" 
    # --training_method "soft_weights" --model_location "model_1" 
    # --features "Variant-All"`
    if args.TRAINING_METHOD not in ["regr_argmaxs", "soft_argmaxs"]:
        print(""" 
        This part was created to test the trained model by feeding it with
        all events and then creating a plot showing the summed
        distribution of the predicted weights.
        """)
        test_on_all_events_1(args)
    else:
        print(""" 
        This part was created to test the trained model by feeding it with
        all events and then creating a plot showing the
        distribution of the predicted alphaCP max.
        """)
        test_on_all_events_2(args)

if args.ACTION == "test_labels":
    # $ python main.py --action "test_labels" --input "data" 
    # --num_classes "51" --features "Variant-All" --hits "hits_c2s"
    print(""" 
    This part was created to double check different labels which are used
    to train the model.
    """)
    test_labels(args)

if args.ACTION == "experimental":
    # exec(open(os.path.normpath("experimental/model_c012s_v1.py")).read())
    # exec(open(os.path.normpath("experimental/model_c012s_v2.py")).read())
    exec(open(os.path.normpath("experimental/model_regr_c012s.py")).read())