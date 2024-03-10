""" This program read the dowloaded original raw data, parse all the records and then
saved them as prepared "rhorho_raw.*.npy: files """
import numpy as np
from prepare_utils import read_raw_asci
import argparse
import os
from pathlib import Path


def read_raw_all(kind, args):
    """ Read the original raw data and use src_py.prepare_utils.read_raw_asci to parse it
    as data and weights ready for being saved as "rhorho_raw.*.npy files". """
    print(f"Reading and parsing the raw data containing {kind}")

    data_path = args.IN
    all_data = []
    all_weights = []

    for letter in ["a"][:args.DATASETS]:
        name = os.path.join(data_path, "pythia.H.rhorho.1M.%s.%s.outTUPLE_labFrame" % (letter, kind))
        print(f"  ==> {letter}, {name}")
        data, weights = read_raw_asci(name, num_particles=7)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    
    return all_data, all_weights


if __name__ == "__main__":
    
    # Command line arguments needed for running the program independently
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default=2, type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", type=Path, help="data path", default="temp_data")
    args = parser.parse_args()
    
    data_path = args.IN
    data_copy = []
    n_events = 0

    for i in range(0, 21):
        if i < 10:
            filename = f"CPmix_0{i}"
        else:
            filename = f"CPmix_{i}"
        
        # Loading data and parsing it to data and weights
        data, weights = read_raw_all(filename, args)

        # Verifying data, as it should be the same for all the CPmix_CLASS_INDEX cases
        if i == 0:
            data_copy = data
            n_events = len(weights)
        np.testing.assert_almost_equal(data_copy, data)

        # Saving the weights
        if i < 10:
            weights_path = f"rhorho_raw.w_0{i}.npy"
        else:
            weights_path = f"rhorho_raw.w_{i}.npy"
        np.save(os.path.join(data_path, weights_path), weights)

    # Preparing permutations for data shuffling
    np.random.seed(123)
    perm = np.random.permutation(n_events)

    # Saving the data and permutations
    np.save(os.path.join(data_path, "rhorho_raw.data.npy"), data_copy)
    np.save(os.path.join(data_path, "rhorho_raw.perm.npy"), perm)

    print(f"In total: prepared {len(weights)} events.")
