import argparse
from data.loaders.data_loader import download_data
from data.preprocessing.cpmix_utils import preprocess_data
from data.preprocessing.data_utils import EventDatasets
from data.preprocessing.rhorho import RhoRhoEvent
import model.rhorho_model as rhorho_model
from pathlib import Path

# =============================== GETTING ALL THE ARGUMENTS ============================================
# Specifiying the model and its function responsible for running the training process
types = {"nn_rhorho": rhorho_model.start}

# Initialising a parser handling all the commaind-line arguments and options
parser = argparse.ArgumentParser(
  prog='Higgs Boson CP Classifier',
  description='Download data and train the classifier for the Higgs Boson CP problem')

# Adding the arguments used by data_loader.py
parser.add_argument("-i", "--input", dest="IN", type=Path, help="data path")
parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", type=bool, 
                    default=False, help="overwriting existing data")

# Adding the arguments used by cpmix_utils.py
parser.add_argument("--num_classes", dest="NUM_CLASSES", type=int, default=0,
                    help="number of classes used for discretisation")
parser.add_argument("--reuse_weights", dest="REUSE_WEIGHTS", type=bool, default=False,
                    help="set this flag to True if you want to reuse the calculated weights")
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho',
                    help="decay mode for training")
parser.add_argument("--ohe_coefficients", dest="OHE_C_COEFFICIENTS", 
                    choices=["one_hot_encoded_c0", "one_hot_encoded_c1",  "one_hot_encoded_c2"], 
                    default="one_hot_encoded_c0",
                    help="?") # TODO: Add a help message

# Unclear to the project team
parser.add_argument("--restrict_most_probable_angle", dest="RESTRICT_MOST_PROBABLE_ANGLE", 
                    type=bool, default=False)
parser.add_argument("--normalize_weights", dest="NORMALIZE_WEIGHTS", type=bool, 
                    default=False)

# Adding the arguments used by data_utils.py
parser.add_argument("--miniset", dest="MINISET", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help="using the small version of the training data set")
parser.add_argument("--unweighted", dest="UNWEIGHTED", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help="\"unweighting\" the events to resemble real data")

# Adding the arguments used by rhorho.py
parser.add_argument("--beta",  type=float, dest="BETA", 
                    help="the beta parameter value for polynomial smearing", default=0.0)
parser.add_argument("-f", "--features", dest="FEAT", help="Features", 
                    choices= ["Variant-All", "Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1",
                              "Variant-2.2", "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"], 
                              default="Variant-All")
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], default="A")

# Adding the arguments used by rhorho_model.py
parser.add_argument("--training_method", dest="TRAINING_METHOD", 
                    choices=["soft_weights", "soft_c_coefficients",  "soft_argmaxes", 
                             "regr_c_coefficients", "regr_weights", "regr_argmaxes"], 
                             default="soft_weights", help="training method (the loss function type)")
parser.add_argument("--plot_features", dest="PLOT_FEATURES", choices=["NO", "FILTER", "NO-FILTER"], 
                    default="NO", help="?") # TODO: Add a help message
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help = "number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.0,
                    help="dropout probability")
parser.add_argument("-o", "--optimizer", dest="OPT", 
                    choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
                            "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
                            "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], 
                    default="AdamOptimizer", help="TensorFlow optimiser")
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3,
                    help="the number of epochs used during the training process")

# Adding the arguments used by tf_model.py
parser.add_argument(
    "--delt_classes", dest="DELT_CLASSES", type=int, default=0, 
    help=("Maximum allowed difference between the predicted class" + 
          "and the true class for an event to be considered correctly classified."))

# Parsing the command-line arguments 
args = parser.parse_args()

# =================================== TRAINING THE MODEL ===============================================
# Calling the main function of the specified model (rhorho model by default)
types[args.TYPE](args)

# TEST (Downloading and preprocessing data, training the model):
# $ python .\main.py --input "data/raw_npy" --num_classes 25 --miniset "yes" --plot_features "FILTER"