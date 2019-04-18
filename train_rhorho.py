import numpy as np
import tensorflow as tf
from scipy import optimize

from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
import os


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

def run(args):
    data_path = args.IN
    num_classes = args.NUM_CLASSES

    print "Loading data"
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w = read_np(os.path.join(data_path, "rhorho_raw.w.npy")).swapaxes(0,1)
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data.shape[0]
    x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]) * np.pi

    data_len = data.shape[0]
    classes = np.linspace(0, 1, num_classes) * np.pi
    weights = np.zeros((data_len, num_classes))
    arg_maxs = np.zeros(data_len)
    reuse_weigths = True  # Set this flag to true if you want reuse calculated weights

    if not os.path.exists(os.path.join(data_path, 'popts.npy')):
        popts = np.zeros((data_len, 3))
        for i in range(data_len ):
            popt, pcov = optimize.curve_fit(weight_fun, x, w[i, :], p0=[1, 1, 1])
            popts[i] = popt

        np.save(os.path.join(data_path, 'popts.npy'), popts)
    popts = np.load(os.path.join(data_path, 'popts.npy'))
    if not reuse_weigths or not os.path.exists(os.path.join(data_path, 'weigths.npy'))\
            or not os.path.exists(os.path.join(data_path, 'arg_maxs.npy')):
        for i in range(data_len):
            weights[i] = weight_fun(classes, *popts[i])
            arg_max = 0
            if weight_fun(2 * np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = 2 * np.pi
            phi = np.arctan(popts[i][2] / popts[i][1])

            if 0 < phi < 2 * np.pi and weight_fun(phi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = phi
            if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
                arg_max = phi + np.pi
            if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *popts[i]) > weight_fun(arg_max,
                                                                                                       *popts[i]):
                arg_max = phi + 2 * np.pi

            arg_maxs[i] = arg_max
        np.save(os.path.join(data_path, 'weigths.npy'), weights)
        np.save(os.path.join(data_path, 'arg_maxs.npy'), arg_maxs)
    weights = np.load(os.path.join(data_path, 'weigths.npy'))
    arg_maxs = np.load(os.path.join(data_path, 'arg_maxs.npy'))
    arg_maxs[arg_maxs>np.pi] = arg_maxs[arg_maxs>np.pi] - np.pi


    print "Processing data"
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, weights, arg_maxs, perm, popts=popts, miniset=args.MINISET, unweighted=args.UNWEIGHTED)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

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