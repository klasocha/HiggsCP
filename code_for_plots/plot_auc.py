from sys import stderr
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

#---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Plots")
parser.add_argument("-i", "--input", dest="IN", required=True, nargs="*")
parser.add_argument("-o", "--output", dest="OUT", required=True)
parser.add_argument("-s", "--sizes", dest="SIZES", nargs=4,
	help="x0 x1 y0 y1; set corners of output plot", type=float, default=[1.0, 25.0, 0.500, 0.830])
parser.add_argument("-t", "--colors_t", dest="COLORS_T", nargs="*")
parser.add_argument("-v", "--colors_v", dest="COLORS_V", nargs="*")
parser.add_argument("-x", "--legendloc", dest="LOC",
	help="set location of the legend on the plot; see: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend", default="upper right")
parser.add_argument("-l", "--labels", dest="LABELS", nargs="*")
parser.add_argument("--only_validation", dest="ONLYVAL", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
args = parser.parse_args()

#---------------------------------------------------------------------

for dir_path, color_t, color_v, label in zip(args.IN, args.COLORS_T, args.COLORS_V, args.LABELS):
	trains = []
	valids = []
	with open(dir_path) as file:
		for line in file:
			try:
				if ("EPOCH" in line):
					data = line.split(" ")
					valids.append(float(data[-1].strip()))
					trains.append(float(data[-4].strip()))
			except:
				pass
	x = np.arange(1,len(valids)+1)
	trains = np.array(trains)
	valids = np.array(valids)
	if not args.ONLYVAL:
		plt.plot(x, trains, color=color_t, linestyle=":", label=" Training AUC")
	plt.plot(x, valids, color=color_v, linestyle="-.", label=" Validation AUC")

plt.axis(args.SIZES)
plt.legend(loc=args.LOC, frameon=False, title = args.LABELS[0])
plt.xlabel('# epochs')
plt.ylabel('AUC')

plt.savefig(args.OUT)
