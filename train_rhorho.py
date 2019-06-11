import numpy as np
import tensorflow as tf
import os, errno

from cpmix_utils import preprocess_data
from download_data import download_data
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
from monit_utils import monit_plots


def run(args):
    num_classes = args.NUM_CLASSES
    if args.WEIGHTS_SUBSET:
        num_classes = len(args.WEIGHTS_SUBSET)

    download_data(args)
    data, weights, arg_maxs, perm, popts = preprocess_data(args)

    print "Processing data"
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, weights, arg_maxs, perm, popts=popts, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Prepared %d features" % num_features
    
    if args.PLOT_FEATURES is not "NO":
        w_a = weights[:, 0]
        w_b = weights[:, num_classes/2]
        monit_plots(args, event, w_a, w_b)

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_classes,
                              num_layers=args.LAYERS, size=args.SIZE,
                              keep_prob=(1-args.DROPOUT), optimizer=args.OPT,
                              tloss=args.TRAINING_METHOD)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_classes,
                               num_layers=args.LAYERS, size=args.SIZE,
                               keep_prob=(1-args.DROPOUT), optimizer=args.OPT,
                               tloss=args.TRAINING_METHOD)

    tf.global_variables_initializer().run()

    print "Training"
    weights_subset_desc = "_WEIGHTS_SUBS" + str(args.WEIGHTS_SUBSET) if args.WEIGHTS_SUBSET else ""
    pathOUT = "monit_npy/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_"\
              + args.PLOT_FEATURES + "_NUM_CLASSES_" + str(args.NUM_CLASSES) + weights_subset_desc + "/"
    if args.TRAINING_METHOD == 'regr_popts':
        pathOUT = "monit_npy/REGRESSION_" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" \
                  + args.PLOT_FEATURES + "/"
    if pathOUT:
        try:
            os.makedirs(pathOUT)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    total_train(pathOUT, model, points, args, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        run(args)


if __name__ == "__main__":
    start(args = {})
