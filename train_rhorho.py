""" This program initialises and trains a neural network model using TensorFlow 
for solving the problems related to the rhorho events analysis """

import numpy as np
import tensorflow.compat.v1 as tf
import os, errno
from src_py.cpmix_utils import preprocess_data
from src_py.download_data_rhorho import download_data
from src_py.rhorho import RhoRhoEvent
from src_py.data_utils import EventDatasets
from src_py.tf_model import total_train, NeuralNetwork
from src_py.monit_utils import monit_plots


def run(args):
    # Getting the command-line arguments
    num_classes = args.NUM_CLASSES

    # ==================================== DATA PREPARATION ============================================
    # TEST: $ python .\main.py --input "data/raw_npy"
    print("\033[1mDownloading data...\033[0m")
    download_data(args)
    
    # TEST: $ python .\main.py --num_classes 25 --type "nn_rhorho" --input "data/raw_npy" 
    print("\033[1mPreprocessing data...\033[0m")
    data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s = preprocess_data(args)
    
    # TEST: python .\main.py --num_classes 25 --type "nn_rhorho" --input "data/raw_npy" 
    # --reuse_weights True --miniset "yes"
    print("\033[1mProcessing data...\033[0m")
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, weights, argmaxs, perm, c012s=c012s, hits_argmaxs=hits_argmaxs,  
                           hits_c012s=hits_c012s, miniset=args.MINISET, unweighted=args.UNWEIGHTED)
    num_features = points.train.x.shape[1]
    print(f"{num_features} features have been prepared.")
    
    # TEST: Input data shape: 
    """    
    print(f"{num_features} features have been prepared.")
    print('x:\n', points.train.x[0:1])
    print('x:\n', points.train.x.shape)
    print('filt:\n', points.train.filt[0:1])
    print('filt:\n', points.train.filt.shape)
    print('weights:\n', points.train.weights[0:1])
    print('weights:\n', points.train.weights.shape)    
    print('argmaxes:\n', points.train.argmaxs[0:1])
    print('argmaxes:\n', points.train.argmaxs.shape)
    print('c_coefficients:\n', points.train.c_coefficients[0:1])
    print('c_coefficients:\n', points.train.c_coefficients.shape)
    print('ohe_argmaxes:\n', points.train.ohe_argmaxes[0:1])
    print('ohe_argmaxes:\n', points.train.ohe_argmaxes.shape)    
    print('ohe_coefficients:\n', points.train.ohe_coefficients[0:1])    
    print('ohe_coefficients:\n', points.train.ohe_coefficients.shape)    
    print('mask:\n', points.train.mask[0:1])
    print('mask:\n', points.train.mask.shape)
    print('n:\n', points.train.n) 
    """

    # =========================== PREPARING FOLDERS FOR STORING THE RESULTS ============================
    pathOUT = "temp_results/"+ args.TYPE + "_" + args.FEAT + "_" + args.TRAINING_METHOD + \
        "_" + args.HITS_C012s + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + \
            args.PLOT_FEATURES + "_NUM_CLASSES_" + str(args.NUM_CLASSES) + "/"
    
    if pathOUT:
        try:
            os.makedirs(pathOUT)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pathOUT_npy = pathOUT+'monit_npy/'
    if pathOUT_npy:
        try:
            os.makedirs(pathOUT_npy)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pathOUT_plots = pathOUT+'monit_plots/'
    if pathOUT_plots:
        try:
            os.makedirs(pathOUT_plots)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # ====================================== OPTIONAL PLOTTING =========================================
    # Plotting the weights distribution (scalar/pseudoscalar)
    if args.PLOT_FEATURES is not "NO":
        w_a = weights[:,0]
        w_b = weights[:,num_classes/2]
        monit_plots(pathOUT_plots, args, event, w_a, w_b)

    # ===================================== MODEL INITIALISATION =======================================
    # Model initialisation
    print("Initializing model")

    with tf.variable_scope("model1"):
        model = NeuralNetwork(num_features, num_classes,
                              num_layers=args.LAYERS, size=args.SIZE,
                              keep_prob=(1-args.DROPOUT), optimizer=args.OPT,
                              tloss=args.TRAINING_METHOD)

    with tf.variable_scope("model1", reuse=True):
        emodel = NeuralNetwork(num_features, num_classes,
                               num_layers=args.LAYERS, size=args.SIZE,
                               keep_prob=(1-args.DROPOUT), optimizer=args.OPT,
                               tloss=args.TRAINING_METHOD)

    tf.global_variables_initializer().run()

    print("Training")
    total_train(pathOUT_npy, model, points, args, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    # Resetting the default TensorFlow graph. It clears the current default graph stack
    # and resets the global default graph:
    tf.reset_default_graph()

    # Creating a new TensorFlow session:   
    sess = tf.Session()

    # Disabling th eager execution mode to allow initialising tf.placeholders
    # (we use tf sessions anyway)
    tf.compat.v1.disable_eager_execution()

    # Setting the seed for the NumPy random number generator to ensure reproducibility:    
    np.random.seed(781)

    # Setting the seed for the TensorFlow random number generator to ensure reproducibility:
    tf.set_random_seed(781)
    
    # Starting a context manager where the TensorFlow session sess is set as the default session 
    # within the block. Inside this context manager, the run(args) function is called 
    # to execute the main logic of the program:    
    with sess.as_default():
        run(args)


if __name__ == "__main__":
    start(args = {})
