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
from plots.results_analysis_1 import draw as results_analysis_1
from plots.results_analysis_2 import draw as results_analysis_2
from plots.results_analysis_3 import draw as results_analysis_3
from plots.results_analysis_4 import draw as results_analysis_4
from plots.results_analysis_5 import draw as results_analysis_5
from tests.test_parsed_data import test_parsed_data, show_example_records
from tests.test_model_on_unwt_events import test_on_unwt_events
from tests.test_labels import test_labels
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
                    help="which coefficients (C0, C1 or C2) to choose as labels")

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

# Adding other arguments (not used for now)
# parser.add_argument("-lambda", "--lambda", type=float, dest="LAMBDA", help="value of lambda parameter", default=0.0)
# parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
# parser.add_argument("--pol_b", type=float, dest="pol_b", help="value of b parameter for polynomial smearing", default=0.0)
# parser.add_argument("--pol_c", type=float, dest="pol_c", help="value of c parameter for polynomial smearing", default=0.0)
# parser.add_argument("--w1", dest="W1")
# parser.add_argument("--w2", dest="W2")

parser.add_argument("--use_unweighted_events", dest="USE_UNWEIGHTED_EVENTS", action="store_true",
                    help="applying the unweighted events for training (Monte Carlo)", default=False)

# Keras & TFv2 arguments
parser.add_argument("--model_location", dest="MODEL_LOCATION", 
                    help='name of the directory in "results/" containing the model state (weights, metadata)')
parser.add_argument("--use_filtered_data", dest="USE_FILTERED_DATA", 
                    help="picking only those vectors having \"pt\" value greater than 20",
                    action="store_true", default=False)

# Plot arguments
plot_types = {"PHISTAR-DISTRIBUTION" : phistar_dist, # Variant-1.1 should be prepared in advance
         "C012S-WEIGHT" : c012s_weight,
         "C012S-DISTRIBUTION" : c012s_dist,
         "WEIGHTS-FOR-EVENT-VIA-C012": weights_with_c012s,
         "UNWEIGHTED-EVENTS-WEIGHTS": unwt_weights,
         "RESULTS_ANALYSIS_1": results_analysis_1, # "soft_weights", "regr_weights"
         "RESULTS_ANALYSIS_2": results_analysis_2, # "soft_c012s"
         "RESULTS_ANALYSIS_3": results_analysis_3, # "soft_argmaxs"
         "RESULTS_ANALYSIS_4": results_analysis_4, # "regr_c012s"
         "RESULTS_ANALYSIS_5": results_analysis_5  # "regr_argmaxs"
         }

parser.add_argument("--output", dest="OUT", help="output path for plots", default="figures")
parser.add_argument("--format", dest="FORMAT", 
                    help='the format of the output plots ("png"/"pdf"/"eps")', default="png")
parser.add_argument("--show", dest="SHOW", action="store_true", 
                    help='use it to display the plots before saving them', default=False)
parser.add_argument("--option", dest="OPTION", choices=plot_types.keys(), default="PHISTAR-DISTRIBUTION",
                    help='specify what script for drawing the plots you want to run')
parser.add_argument("--hypothesis", dest="HYPOTHESIS", default="None", 
                    help="Hypothesis: the alphaCP class (e.g. 02)")
parser.add_argument("--dataset", dest="DATASET", help="dataset (train/valid/test)")

# Test arguments
parser.add_argument("--source-1", dest="SOURCE_1",
                    help="the first directory containing data to be compared")
parser.add_argument("--source-2", dest="SOURCE_2", 
                    help="the second directory containing data to be compared")
parser.add_argument("--datasets", dest="DATASETS", default=2, type=int, help="number of datasets to prepare")

# Main controller
parser.add_argument("--action", dest="ACTION", choices=["download_and_prepare_original", "download_and_preprocess",  
                    "train", "continue_training", "predict_train_and_valid", "plot", "test_parsed_data", 
                    "test_model_on_unwt_events", "predict_test", "test_labels"], 
                    default="train")
parser.add_argument("--keras", dest="KERAS", choices=["v2", "v3"], default="v3", help="the version of the Keras engine")

# Parsing the command-line arguments 
args = parser.parse_args()

# =================================== CONTROLING THE ML FLOW  ==========================================
if args.ACTION == "download_and_prepare_original":
    # $ python main.py --action "download_and_prepare_original" --input "data_original"
    download_original_data(args)
    prepare_rhorho(args)

if args.ACTION == "download_and_preprocess":
    # $ python main.py --action "download_and_preprocess" --input "data" --features Variant-All
    # --num_classes "11"
    prepare_data(args)

if args.ACTION in ["train", "continue_training", "predict_train_and_valid", "predict_test"]:
    # 1. python main.py --action "train" --input "data" --num_classes "11" --epochs "2"
    # --training_method "soft_weights" --model_location "model_1"
    
    # 2. python main.py --action "continue_training" --input "data" --num_classes "11"
    # --epochs "3" --training_method "soft_weights" --model_location "model_1"
    
    # 3. python main.py --action "predict_train_and_valid" --input "data" --num_classes "11"
    # --model_location "model_1"
    
    # 4. python main.py --action "predict_test" --input "data" --num_classes "11"
    # --model_location "model_1"
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
    # --output "plots/figures/test_model_on_unwt_events" --num_classes "21" 
    # --hypothesis "0" --training_method "soft_weights" --model_location "model_1" 
    # --features "Variant-All"
    print(""" 
    This part was created to test the trained model by feeding it with
    the unweighted events and the creating a plot showing the summed
    distribution of the predicted weights.
    """)
    test_on_unwt_events(args)

if args.ACTION == "test_labels":
    # $ python main.py --action "test_labels" --input "data" 
    # --num_classes "51" --features "Variant-All" --hits "hits_c2s"
    print(""" 
    This part was created to double check different labels which are used
    to train the model.
    """)
    test_labels(args)