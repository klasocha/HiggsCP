import argparse
import os

import train_rhorho

types = {"nn_rhorho": train_rhorho.start}

parser = argparse.ArgumentParser(description='Train classifier')
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho')
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help="number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-lambda", "--lambda", type=float, dest="LAMBDA", help="value of lambda parameter", default=0.0)
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], default="A")
parser.add_argument("-o", "--optimizer", dest="OPT",
                    choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
                             "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
                             "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], default="AdamOptimizer")
parser.add_argument("-i", "--input", dest="IN", default=os.environ["RHORHO_DATA"])
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.2)
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=3)
parser.add_argument("-f", "--features", dest="FEAT", help="Features",
                    choices=["Variant-All", "Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1",
                             "Variant-2.2", "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"],
                    default="Variant-All")
parser.add_argument("--miniset", dest="MINISET", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
parser.add_argument("--num_classes", dest="NUM_CLASSES", type=int, default=21)
parser.add_argument("--unweighted", dest="UNWEIGHTED", type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=False)
parser.add_argument("--reuse_weights", dest="REUSE_WEIGTHS", type=bool, default=False,
                    help='Forces to reuse calculated weights (when available)')
parser.add_argument("--restrict_most_probable_angle", dest="RESTRICT_MOST_PROBABLE_ANGLE", type=bool, default=False,
                    help='Restricts range of most probable mixing angle from (0,2pi) to (0,pi)')
parser.add_argument("--force_download", dest="FORCE_DOWNLOAD", type=bool, default=False,
                    help='When set, it forces to download data from server')
parser.add_argument("--normalize_weights", dest="NORMALIZE_WEIGHTS", type=bool, default=False,
                    help='Normalize weights to make constant term equal one')

parser.add_argument("--data_class_distance", dest="DATA_CLASS_DISTANCE", type=int, default=0,
                    help='Maximal distance between predicted and valid class when event is found as correctly classified')

parser.add_argument("--data_url", dest="DATA_URL",
                    default='http://th-www.if.uj.edu.pl/~erichter/forMichal/HiggsCP_data_CPmix/', type=str,
                    help='set url to data location (with files `rhorho_raw.data.npy`, `rhorho_raw.perm.npy`, `rhorho_raw.w_<i>.npy`)')

parser.add_argument("--weights_subset", dest="WEIGHTS_SUBSET", default=None, type=list,
                    help="If set reduce number of weights to given indexes")
parser.add_argument("--beta", type=float, dest="BETA", help="value of beta parameter for polynomial smearing",
                    default=0.0)
parser.add_argument("--pol_b", type=float, dest="pol_b", help="value of b parameter for polynomial smearing",
                    default=0.0)
parser.add_argument("--pol_c", type=float, dest="pol_c", help="value of c parameter for polynomial smearing",
                    default=0.0)

parser.add_argument("--w1", dest="W1")
parser.add_argument("--w2", dest="W2")

parser.add_argument("--plot_features", dest="PLOT_FEATURES", choices=["NO", "FILTER", "NO-FILTER"], default="NO")

args = parser.parse_args()

types[args.TYPE](args)
