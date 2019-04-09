import argparse

import train_rhorho, train_rhorho_CPmix, train_a1rho, train_a1a1, train_boostedtrees, train_svm, train_randomforest, train_rhorhoZ, train_a1rhoZ, train_a1a1Z 

types = {"nn_rhorho": train_rhorho.start, "nn_rhorho_CPmix": train_rhorho_CPmix.start, "nn_a1rho": train_a1rho.start, "nn_a1a1": train_a1a1.start, 
"boosted_trees": train_boostedtrees.start, "svm": train_svm.start, "random_forest": train_randomforest.start,
         "nn_rhorhoZ": train_rhorhoZ.start, "nn_a1rhoZ": train_a1rhoZ.start, "nn_a1a1Z": train_a1a1.start  }

parser = argparse.ArgumentParser(description='Train classifier')
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho')
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help = "number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-m", "--method", dest="METHOD", choices=["A", "B", "C"], default="A")
parser.add_argument("-o", "--optimizer", dest="OPT", 
	choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
	 "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
	 "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], default="AdamOptimizer")
parser.add_argument("-i", "--input", dest="IN", required=True)
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.2)
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=25)
parser.add_argument("-f", "--features", dest="FEAT", help="Features",
	choices= ["Variant-All", "Variant-1.0", "Variant-1.1", "Variant-2.0", "Variant-2.1", "Variant-2.2", "Variant-3.0", "Variant-3.1", "Variant-4.0", "Variant-4.1"])
parser.add_argument("--treedepth", dest="TREEDEPTH", type=int, default=5)
parser.add_argument("--miniset", dest="MINISET", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
parser.add_argument("--svm_c", dest="SVM_C", type=float)
parser.add_argument("--svm_gamma", dest="SVM_GAMMA", type=float)
parser.add_argument("--forest_max_feat", dest="FOREST_MAX_FEAT", choices=["log2", "sqrt"], default="sqrt")
parser.add_argument("--forest_max_depth", dest="FOREST_MAX_DEPTH", default=10, type=int)
parser.add_argument("--forest_estimators", dest="FOREST_ESTIMATORS", default=10, type=int)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
parser.add_argument("--unweighted", dest="UNWEIGHTED", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)

parser.add_argument("--beta", type=float, dest="BETA", help="value of beta parameter for polynomial smearing", default=0.0)
parser.add_argument("--pol_b", type=float, dest="pol_b", help="value of b parameter for polynomial smearing", default=0.0)
parser.add_argument("--pol_c", type=float, dest="pol_c", help="value of c parameter for polynomial smearing", default=0.0)

parser.add_argument("--w1", dest="W1")
parser.add_argument("--w2", dest="W2")

parser.add_argument("--plot_features", dest="PLOT_FEATURES", choices=["NO", "FILTER", "NO-FILTER"], default="NO")

args = parser.parse_args()

types[args.TYPE](args)
