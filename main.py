import argparse
import os
from utilities.download_original_data import download as download_original_data
from utilities.tf_model import run as train_model
from utilities.prepare_data import prepare_data
from plots.plot_phistar_distribution import draw as phistar_dist 
from plots.plot_popts_rhorho import draw as c012s_weight
from plots.plot_calc_c012s import draw as c012s_dist
from plots.plot_weights_with_c012s import draw as weights_with_c012s
from plots.plot_unwt_weights import draw as unwt_weights
from tests.test_data import test_parsed_data, show_example_records
from utilities.prepare_rhorho import prepare_rhorho

# =============================== GETTING ALL THE ARGUMENTS ============================================
# Initialising a parser handling all the commaind-line arguments and options
parser = argparse.ArgumentParser(
  prog='Higgs Boson CP Classifier',
  description='Download data and train the classifier for the Higgs Boson CP problem')

# Adding the arguments used by src_py/download_data_rhorho.py
parser.add_argument("-i", "--input", dest="IN", type=os.fspath, help="data path", default="temp_data")
parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", action="store_true", 
                    default=False, help="overwriting existing data")

# Adding the arguments used by src_py/cpmix_utils.py
parser.add_argument("--num_classes", dest="NUM_CLASSES", type=int, default=0,
                    help="number of classes used for discretisation")
parser.add_argument("--reuse_weights", dest="REUSE_WEIGHTS", action="store_true", default=False,
                    help="set this flag to True if you want to reuse the calculated weights")
parser.add_argument("--hits_c012s", dest="HITS_C012s", 
                    choices=["hits_c0s", "hits_c1s",  "hits_c2s"], default="hits_c0s",
                    help="?") # TODO: Add a help message

# TODO: Those two have been so far unclear to the project team
parser.add_argument("--restrict_most_probable_angle", dest="RESTRICT_MOST_PROBABLE_ANGLE", 
                    action="store_true", default=False)
parser.add_argument("--normalize_weights", dest="NORMALIZE_WEIGHTS", action="store_true", 
                    default=False)

# Adding the arguments used by src_py/data_utils.py
parser.add_argument("--miniset", dest="MINISET", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help="using the small version of the training data set")

# Adding the arguments used by src_py/rhorho.py
parser.add_argument("--beta",  type=float, dest="BETA", 
                    help="the beta parameter value for polynomial smearing", default=0.0)
parser.add_argument("-f", "--features", dest="FEAT", help="Features", 
                    choices= ["Variant-All", "Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1",
                              "Variant-2.2", "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"], 
                              default="Variant-All")
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], default="A")

# Adding the arguments used by src_py/tf_model.py
parser.add_argument("--training_method", dest="TRAINING_METHOD", 
                    choices=["soft_weights", "soft_c012s",  "soft_argmaxs", "regr_c012s", "regr_weights", "regr_argmaxs"], 
                    default="soft_weights", help="training method (the loss function type)")
parser.add_argument("--plot_features", dest="PLOT_FEATURES", choices=["NO", "FILTER", "NO-FILTER"], 
                    default="NO", help="?") # TODO: Add a help message
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help = "number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.0,
                    help="dropout probability (applied during the training process)")
parser.add_argument("-o", "--optimizer", dest="OPT", 
                    choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
                            "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
                            "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], 
                    default="AdamOptimizer", help="TensorFlow optimiser")
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3,
                    help="the number of epochs used during the training process")
parser.add_argument("--delt_classes", dest="DELT_CLASSES", type=int, default=0, 
                    help=("maximum allowed difference between the predicted class" + 
                    "and the true class for an event to be considered correctly classified."))

# Adding the arguments downloading the original data
parser.add_argument("--download_original_data", dest="DOWNLOAD_ORIGINAL", help="downloading the original data",
                    action="store_true", default=False)

# Adding other arguments
parser.add_argument("-lambda", "--lambda", type=float, dest="LAMBDA", help="value of lambda parameter", default=0.0)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5) # TODO find out the purpose of this argument
parser.add_argument("--pol_b", type=float, dest="pol_b", help="value of b parameter for polynomial smearing", default=0.0)
parser.add_argument("--pol_c", type=float, dest="pol_c", help="value of c parameter for polynomial smearing", default=0.0)
parser.add_argument("--w1", dest="W1")
parser.add_argument("--w2", dest="W2")
parser.add_argument("--use_unweighted_events", dest="USE_UNWEIGHTED_EVENTS", action="store_true",
                    help="applying the unweighted events for training (Monte Carlo)", default=False)

# Keras & TFv2 arguments
parser.add_argument("--model_location", dest="MODEL_LOCATION", 
                    help='name of the directory in "results/" containing the model state (weights, metadata)')

# Plot arguments
plot_types = {"PHISTAR-DISTRIBUTION" : phistar_dist, # Variant-1.1 should be prepared in advance
         "C012S-WEIGHT" : c012s_weight,
         "C012S-DISTRIBUTION" : c012s_dist,
         "WEIGHTS-FOR-EVENT-VIA-C012": weights_with_c012s,
         "UNWEIGHTED-EVENTS-WEIGHTS": unwt_weights}

parser.add_argument("--output", dest="OUT", help="output path for plots", default="figures")
parser.add_argument("--format", dest="FORMAT", 
                    help='the format of the output plots ("png"/"pdf"/"eps")', default="png")
parser.add_argument("--show", dest="SHOW", action="store_true", 
                    help='use it to display the plots before saving them', default=False)
parser.add_argument("--option", dest="OPTION", choices=plot_types.keys(), default="PHISTAR-DISTRIBUTION",
                    help='specify what script for drawing the plots you want to run')
parser.add_argument("--hypothesis", dest="HYPOTHESIS", default="None", 
                    help="Hypothesis: the alphaCP class (e.g. 02)")

# Test arguments
parser.add_argument("--source-1", dest="SOURCE_1",
                    help="the first directory containing data to be compared")
parser.add_argument("--source-2", dest="SOURCE_2", 
                    help="the second directory containing data to be compared")
parser.add_argument("--datasets", dest="DATASETS", default=2, type=int, help="number of datasets to prepare")

# Main controller
parser.add_argument("--action", dest="ACTION", choices=["download_and_prepare_original", "download_and_preprocess",  
                    "train", "continue_training", "predict", "plot", "test"], default="train")

# Parsing the command-line arguments 
args = parser.parse_args()

# =================================== CONTROLING THE ML FLOW  ==========================================
if args.ACTION == "download_and_prepare_original":
    # $ python main.py --action "download_and_prepare_original" --input "data_original"
    download_original_data(args)
    prepare_rhorho(args)

if args.ACTION == "download_and_preprocess":
    # $ python main.py --action "download_and_preprocess" --input "data" --features Variant-All --num_classes 11
    prepare_data(args)

if args.ACTION in ["train", "continue_training", "predict"]:
    # 1. python main.py --action "train" --input "data" --num_classes "11" --epochs "2" --training_method "soft_weights" --model_location "model_1"
    # 2. python main.py --action "continue_training" --input "data" --num_classes "11" --epochs "3" --training_method "soft_weights" --model_location "model_1"
    # 3. python main.py --action "predict" --input "data" --num_classes "11" --model_location "model_1"
    train_model(args)

if args.ACTION == "plot":
    # Instructions are in the modules located in plots/
    plot_types[args.OPTION](args)

if args.ACTION == "test":
    # $ python main.py --action "test" --source-1 "data" --source-2 "data_original" --input "data_original"
    print(""" 
    This part was created to test 
        1. "prepare_utils.py", 
        2. "prepare_rhorho.py", 
        3. "download_data_rhorho.py"
    """)
    test_parsed_data(args)
    show_example_records(args)