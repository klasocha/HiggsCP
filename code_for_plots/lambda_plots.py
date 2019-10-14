import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Lambda plots')
parser.add_argument("-i", "--input", dest="IN", required=True, nargs='*')
parser.add_argument("-o", "--output", dest="OUT", required=True)
parser.add_argument("-c", "--colors", dest="COLORS", required=True, nargs="*")
parser.add_argument("-m", "--markers", dest="MARKERS", required=True, nargs="*")
parser.add_argument("-l", "--labels", dest="LABELS", nargs="*")

args = parser.parse_args()

results = {}

zero_vals = [0.771, 0.749, 0.728]
mode_i = 0
for dir_path, color, label, marker in zip(args.IN, args.COLORS, args.LABELS, args.MARKERS):
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename), 'r') as f:
            data = f.read().split("\n")
            x = float(filename.split("_")[-1].split(".")[0])/10.0 + 0.1
            print(data[-2])	
            y = float(data[-2])
            results[x] = y

    xs = [k for k, v in sorted(results.iteritems())]
    ys = [v for k, v in sorted(results.iteritems())]

    plt.plot([0] + xs, zero_vals[mode_i:mode_i+1] + ys, color=color, label=label, marker = marker)
    mode_i += 1

plt.legend(loc='upper right')
plt.xlabel(r'$Smearing  \ Param. \ \beta$')
plt.ylabel('AUC')
plt.savefig(args.OUT)
