from sys import stderr
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

#---------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Plots")
parser.add_argument("-i1", "--input1", dest="IN1", required=True, nargs="*")
parser.add_argument("-i2", "--input2", dest="IN2", required=True, nargs="*")
parser.add_argument("-o", "--output", dest="OUT", required=True)
parser.add_argument("-s", "--sizes", dest="SIZES", nargs=4,
	help="x0 x1 y0 y1; set corners of output plot", type=float, default=[1.0, 50.0, 0.500, 0.830])
parser.add_argument("-t", "--colors_t", dest="COLORS_T", nargs="*")
parser.add_argument("-v", "--colors_v", dest="COLORS_V", nargs="*")
parser.add_argument("-x", "--legendloc", dest="LOC",
	help="set location of the legend on the plot; see: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend", default="upper right")
parser.add_argument("-l", "--labels", dest="LABELS", nargs="*")
parser.add_argument("--only_validation", dest="ONLYVAL", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False)
args = parser.parse_args()

#---------------------------------------------------------------------

for dir_path1, dir_path2, color_t, color_v, label in zip(args.IN1, args.IN2, args.COLORS_T, args.COLORS_V, args.LABELS):
	trains = []
	valids = []
	with open(dir_path1) as file:
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
		plt.plot(x, trains, color="C0", linestyle=":", label=args.LABELS[0] + " Training AUC")
	plt.plot(x, valids, color="blue", linestyle="--", label=args.LABELS[0] + " Validation AUC")

	trains = []
	valids = []
	with open(dir_path2) as file:
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
		plt.plot(x, trains, color="C1", linestyle="-.", label=args.LABELS[1] + " Training AUC")
	plt.plot(x, valids, color="red", linestyle="-", label=args.LABELS[1] + " Validation AUC")

plt.axis(args.SIZES)
plt.legend(loc=args.LOC, frameon=False)
plt.xlabel('# epochs')
plt.ylabel('AUC')
plt.title(r"$a_1^{\pm} - \rho^{\mp}$")

plt.savefig(args.OUT)
