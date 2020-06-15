import tensorflow as tf
from rhorho import RhoRhoEvent
from data_utils import read_np, EventDatasets
from tf_model import total_train, NeuralNetwork
import os, errno


def run(args):
    data_path = args.IN

    print "Loading data"
    if args.Zsample == "z65":
        data = read_np(os.path.join(data_path, "rhorho_Higgs_Z_raw.data.npy"))
        w_a  = read_np(os.path.join(data_path, "rhorho_Higgs_Z_raw.w_a.npy"))
        w_b  = read_np(os.path.join(data_path, "rhorho_Higgs_Z_raw.w_b.npy"))
        perm = read_np(os.path.join(data_path, "rhorho_Higgs_Z_raw.perm.npy"))
    if args.Zsample == "z115":
        data = read_np(os.path.join(data_path, "rhorho_Higgs_Z115_raw.data.npy"))
        w_a  = read_np(os.path.join(data_path, "rhorho_Higgs_Z115_raw.w_a.npy"))
        w_b  = read_np(os.path.join(data_path, "rhorho_Higgs_Z115_raw.w_b.npy"))
        perm = read_np(os.path.join(data_path, "rhorho_Higgs_Z115_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    print "Processing data"
    event = RhoRhoEvent(data, args)
    points = EventDatasets(event, w_a, w_b, perm, miniset=args.MINISET, unweighted=args.UNWEIGHTED, filtered=True)

    num_features = points.train.x.shape[1]
    print "Generated %d features" % num_features

    print "Initializing model"
    with tf.variable_scope("model1") as vs:
        model = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    with tf.variable_scope("model1", reuse=True) as vs:
        emodel = NeuralNetwork(num_features, num_layers=args.LAYERS, size=args.SIZE, keep_prob=(1-args.DROPOUT), optimizer=args.OPT)

    tf.global_variables_initializer().run()

    pathOUT = "temp_results/"+ args.TYPE + "_" + args.Zsample + "_" + args.FEAT + "/"
    if pathOUT:
        try:
            os.makedirs(pathOUT)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print "Training"
    total_train(pathOUT, model, points, emodel=emodel, batch_size=128, epochs=args.EPOCHS, metric = args.METRIC)


def start(args):
    sess = tf.Session()
    with sess.as_default():
        run(args)

if __name__ == "__main__":
    start(args = {})
