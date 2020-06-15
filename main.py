import argparse

import train_rhorho, train_rhorhoZ, train_higgs_z_rhorho

types = {"nn_rhorho": train_rhorho.start,
"nn_rhorhoZ": train_rhorhoZ.start, "nn_higgs_z_rhorho": train_higgs_z_rhorho.start}

parser = argparse.ArgumentParser(description='Train classifier')
parser.add_argument("-t", "--type", dest="TYPE", choices=types.keys(), default='nn_rhorho')
parser.add_argument("-l", "--layers", dest="LAYERS", type=int, help = "number of NN layers", default=6)
parser.add_argument("-s", "--size", dest="SIZE", type=int, help="NN size", default=100)
parser.add_argument("-o", "--optimizer", dest="OPT", 
	choices=["GradientDescentOptimizer", "AdadeltaOptimizer", "AdagradOptimizer",
	 "ProximalAdagradOptimizer", "AdamOptimizer", "FtrlOptimizer",
	 "ProximalGradientDescentOptimizer", "RMSPropOptimizer"], default="AdamOptimizer")
parser.add_argument("-i", "--input", dest="IN", required=True)
parser.add_argument("-d", "--dropout", dest="DROPOUT", type=float, default=0.0)
parser.add_argument("-e", "--epochs", dest="EPOCHS", type=int, default=25)
parser.add_argument("-f", "--features", dest="FEAT", help="Features",
	choices= ["Variant-All", "Variant-1.0", "Variant-5.0", "Variant-5.1"])
parser.add_argument("--miniset", dest="MINISET", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
parser.add_argument("--z_noise_fraction", dest="Z_NOISE_FRACTION", type=float, default=0.5)
parser.add_argument("--unweighted", dest="UNWEIGHTED", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)

parser.add_argument("--metric", dest="METRIC",  choices=["roc_auc", "prec_score"], default="roc_auc")
parser.add_argument("--Zsample", dest="Zsample",  choices=["z65", "z115"], default="z65")

args = parser.parse_args()

types[args.TYPE](args)
