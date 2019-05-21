import numpy as np
import tensorflow as tf

from cpmix_utils import preprocess_data
from download_data import download_data
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork



def run(args):
    num_classes = args.NUM_CLASSES

    download_data(args)
    data, weights, arg_maxs, perm, popts = preprocess_data(args)

    print "Processing data"
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, weights, arg_maxs, perm, popts=popts, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Prepared %d features" % num_features

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_classes, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print "Training"
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS)


def start(args):
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        run(args)


if __name__ == "__main__":
    start(args = {})
