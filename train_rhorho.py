import tensorflow as tf
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from plot_utils import feature_plot
from tf_model import total_train, NeuralNetwork
import os
import numpy as np


def run(args):
    data_path = args.IN

    print "Loading data"
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w_a = read_np(os.path.join(data_path, "rhorho_raw.w_a.npy"))
    w_b = read_np(os.path.join(data_path, "rhorho_raw.w_b.npy"))
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    print "Processing data"
    event = RhoRhoEvent(data, args)


    points = EventDatasets(event, w_a, w_b, perm, miniset=args.MINISET, unweighted=args.UNWEIGHTED, smear_polynomial=(args.BETA>0), filtered=True)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

    if args.PLOT_FEATURES == "FILTER":
        for i in range(num_features):
            feature_plot(event.cols[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = event.labels[i], w_a = w_a, w_b = w_b, filt = event.cols[:,-1])
            #FIXME
            #feature_plot(event.cols_suppl[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
            #             filename = event.labels_suppl[i], w_a = w_a, w_b = w_b, filt = event.cols_suppl[:,-1])
    elif args.PLOT_FEATURES == "NO-FILTER":
        for i in range(num_features):
            feature_plot(event.cols[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
                         filename = event.labels[i], w_a = w_a, w_b = w_b)
            #FIXME
            #feature_plot(event.cols_suppl[:,i], directory = "../monit_plots/" + args.TYPE + "_" + args.FEAT + "_Unweighted_" + str(args.UNWEIGHTED) + "_" + args.PLOT_FEATURES + "/",
            #             filename = event.labels_suppl[i], w_a = w_a, w_b = w_b)


    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})
