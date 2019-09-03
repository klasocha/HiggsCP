import numpy as np
from prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(kind, args):
    print "Reading %s" % kind

    data_path = args.IN

    all_data = []
    all_weights = []
    for letter in ["a","b","c","d","e","f","g","h","i","k"][:args.DATASETS]:
        name = os.path.join(data_path, "pythia.H.a1a1.1M.%s.%s.outTUPLE_labFrame" % (letter, kind))
        print letter, name
        data, weights = read_raw_root(name, num_particles=9)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default=10, type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN")
    args = parser.parse_args()

    data_path = args.IN

    for i in range(0, 21, 2):
        data, weights = read_raw_all("CPmix_%s" % str(i).zfill(2), args)
        if i == 0:
            data_00 = data
            np.random.seed(123)
            perm = np.random.permutation(len(weights))
            np.save(os.path.join(data_path, "a1a1_raw.data.npy"), data_00)
            np.save(os.path.join(data_path, "a1a1_raw.perm.npy"), perm)
        else:
            np.testing.assert_array_almost_equal(data_00, data)
        np.save(os.path.join(data_path, "a1a1_raw.w_%s.npy" % str(i).zfill(2)), weights)
