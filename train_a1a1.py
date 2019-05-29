import numpy as np
import tensorflow as tf
import os, errno

from cpmix_utils_a1a1 import preprocess_data
from download_data_a1a1 import download_data
from a1a1 import A1A1Event
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
from monit_utils import monit_plots
from plot_utils import plot_one_TH1D, plot_two_TH1D


def run(args):
    num_classes = args.NUM_CLASSES

    print "Loading data"
    download_data(args)
    data, weights, arg_maxs, perm, popts = preprocess_data(args)
    data_path = args.IN

    print "Processing data"
    event = A1A1Event(data, args)
    points = EventDatasets(event, weights, arg_maxs, perm, popts=popts, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Prepared %d features" % num_features

    if args.PLOT_FEATURES is not "NO":
        w_a = weights[:,0]
        w_b = weights[:,num_classes/2]
        monit_plots(args, event, w_a, w_b)
   
    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    pathOUT = "slurm_results/monit_npy/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "_NUM_CLASSES_" + str(args.NUM_CLASSES) + "/"
    if pathOUT:
        try:
            os.makedirs(pathOUT)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    total_train(pathOUT, model, points, args, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})
