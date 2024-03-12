"""
This program runs the scripts available in plot_py package
"""
import argparse
from pathlib import Path
from plot_py.plot_phistar_distribution import draw as phistar_dist 
from plot_py.plot_popts_rhorho import draw as c012s_weight
from plot_py.plot_calc_c012s import draw as c012s_dist

# Command line arguments needed for running the program independently
types = {"PHISTAR-DISTRIBUTION" : phistar_dist, # Variant-1.1 should be prepared in advance
         "C012S-WEIGHT" : c012s_weight,
         "C012S-DISTRIBUTION" : c012s_dist}
parser = argparse.ArgumentParser(description='Diagram creator')
parser.add_argument("-i", "--input", dest="IN", type=Path, help="input data path", default="../temp_data")
parser.add_argument("-o", "--output", dest="OUT", type=Path, help="output path for diagrams", default="figures")
parser.add_argument("-f", "--format", dest="FORMAT", 
                    help='the format of the output diagrams ("png"/"pdf"/"eps")', default="png")
parser.add_argument("-s", "--show", dest="SHOW", type=bool, 
                    help='set to False to save the diagrams without showing them', default=True)
parser.add_argument("--option", dest="OPTION", choices=types.keys(), default="PHISTAR-DISTRIBUTION",
                    help='specify what script for drawing the diagrams you want to run', required=True)
parser.add_argument("--hypothesis", dest="HYPOTHESIS", default="None", 
                    help="Hypothesis: the alphaCP class (e.g. 02)")
args = parser.parse_args()
types[args.OPTION](args)