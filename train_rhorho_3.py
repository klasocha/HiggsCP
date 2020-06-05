import tensorflow as tf
from rhorho_3 import RhoRhoEvent
from data_utils_3 import read_np, EventDatasets
from tf_model_3 import total_train, NeuralNetwork
import numpy as np
import os


def run(args):
    data_path = args.IN

    # I changed code below according to instructions in email
    # print("Loading data")
    # data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    # w_a = read_np(os.path.join(data_path, "rhorho_raw.w_a.npy"))
    # w_b = np.zeros(w_a.shape)
    # perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w_a = read_np(os.path.join(data_path, "rhorho_raw.w_a.npy"))
    zeros = np.zeros(w_a.shape)
    w_a = np.concatenate((w_a,zeros),axis = 0)
    w_b = np.concatenate((zeros,read_np(os.path.join(data_path, "Z_65_155.rhorho_raw.w_a.npy"))),axis = 0)
    perm = np.concatenate((read_np(os.path.join(data_path, "rhorho_raw.perm.npy")),read_np(os.path.join(data_path, "Z_65_155.rhorho_raw.perm.npy"))))
    print("Read %d events" % data.shape[0])

    print("Processing data")
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, w_a, w_b, perm, miniset=args.MINISET, unweighted=args.UNWEIGHTED, smear_polynomial=(args.BETA>0), filtered=True)

    num_features = points.train.x.shape[1]
    print("Generated %d features" % num_features)

    print("Initializing model")
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    print("Training")
    total_train(model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS, metric = args.METRIC)


def start(args):
    sess = tf.compat.v1.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})
