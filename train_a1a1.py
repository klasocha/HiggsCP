import errno
import os

import tensorflow as tf

from src_py.a1a1 import A1A1Event
from src_py.cpmix_utils import preprocess_data
from src_py.data_utils import EventDatasets
from src_py.monit_utils import monit_plots
from src_py.tf_model import total_train, NeuralNetwork


def run(args):
    num_classes = args.NUM_CLASSES

    print "Loading data"
    data, weights, arg_maxs, perm, popts = preprocess_data(args)

    print "Processing data"
    event = A1A1Event(data, args)
    points = EventDatasets(event, weights, arg_maxs, perm, popts=popts, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Prepared %d features" % num_features

    pathOUT = os.path.join("temp_results", args.TYPE + "_" + args.FEAT + "_" + args.TRAINING_METHOD + \
                           "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "_NUM_CLASSES_" + \
                           str(args.NUM_CLASSES))

    if args.PLOT_FEATURES:
        w_a = weights[:,0]
        w_b = weights[:,num_classes/2]
        monit_plots(os.path.join(pathOUT, 'monit_plots'), args, event, w_a, w_b)
   
    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    total_train(os.path.join(pathOUT, "monit_npy"), model, points, args, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)
