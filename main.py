import argparse
import os
import train_rhorho, train_a1rho, train_a1a1
from pathlib import Path

# =============================== GETTING ALL THE ARGUMENTS ============================================
# Specifiying the model and its function responsible for running the training process
types = {"nn_rhorho": train_rhorho.start,"nn_a1rho": train_a1rho.start,"nn_a1a1": train_a1a1.start}

# Initialising a parser handling all the commaind-line arguments and options
parser = argparse.ArgumentParser(
  prog='Higgs Boson CP Classifier',
  description='Download data and train the classifier for the Higgs Boson CP problem')

# Adding the arguments used by src_py/download_data_rhorho.py
parser.add_argument("-i", "--input", dest="IN", type=Path, help="data path")
parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", type=bool, 
                    default=False, help="overwriting existing data")

# Adding the arguments used by src_py/cpmix_utils.py
parser.add_argument("--num_classes", dest="NUM_CLASSES", type=int, default=0,
                    help="number of classes used for discretisation")
parser.add_argument("--reuse_weights", dest="REUSE_WEIGHTS", type=bool, default=False,
                    help="set this flag to True if you want to reuse the calculated weights")
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho',
                    help="decay mode for training")
parser.add_argument("--hits_c012s", dest="HITS_C012s", 
                    choices=["hits_c0s", "hits_c1s",  "hits_c2s"], default="hits_c0s",
                    help="?") # TODO: Add a help message

# TODO: Those two have been so far unclear to the project team
parser.add_argument("--restrict_most_probable_angle", dest="RESTRICT_MOST_PROBABLE_ANGLE", 
                    type=bool, default=False)
parser.add_argument("--normalize_weights", dest="NORMALIZE_WEIGHTS", type=bool, 
                    default=False)

# Adding the arguments used by src_py/data_utils.py
parser.add_argument("--miniset", dest="MINISET", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help="using the small version of the training data set")
parser.add_argument("--unweighted", dest="UNWEIGHTED", 
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False,
                    help="\"unweighting\" the events to resemble real data")

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
                    help="dropout probability")
parser.add_argument("-o", "--optimizer", dest="OPT", 
                    choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
                            "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
                            "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], 
                    default="AdamOptimizer", help="TensorFlow optimiser")
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3,
                    help="the number of epochs used during the training process")
parser.add_argument("--delt_classes", dest="DELT_CLASSES", type=int, default=0, 
                    help=("Maximum allowed difference between the predicted class" + 
                    "and the true class for an event to be considered correctly classified."))

# Adding other arguments
parser.add_argument("-lambda", "--lambda", type=float, dest="LAMBDA", help="value of lambda parameter", default=0.0)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
parser.add_argument("--pol_b", type=float, dest="pol_b", help="value of b parameter for polynomial smearing", default=0.0)
parser.add_argument("--pol_c", type=float, dest="pol_c", help="value of c parameter for polynomial smearing", default=0.0)
parser.add_argument("--w1", dest="W1")
parser.add_argument("--w2", dest="W2")
parser.add_argument("--hits_c012s", dest="HITS_C012s", choices=["hits_c0s", "hits_c1s",  "hits_c2s"], default="hits_c0s")

# Parsing the command-line arguments 
args = parser.parse_args()

# =================================== TRAINING THE MODEL ===============================================
# Calling the main function of the specified model (rhorho model by default)
types[args.TYPE](args)

# TEST (Downloading and preprocessing data, training the model):
# $ python .\main.py --input "data" --type nn_rhorho --epochs 5 --features Variant-All --num_classes 11