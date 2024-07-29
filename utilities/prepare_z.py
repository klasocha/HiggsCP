""" This program read the dowloaded original raw data, parse all the records and then
saved them as prepared "rhorho_raw.*.npy: files """
import numpy as np
from .prepare_utils import read_raw_asci
import os


def read_raw_all(args, kind=None):
    """ Read the original raw data and use src_py.prepare_utils.read_raw_asci to parse it
    as data and weights ready for being saved as "rhorho_raw.*.npy files". """
    print(f"Reading and parsing the raw data containing {kind}")

    data_path = args.IN
    all_data = []
    all_weights = []

    for letter in ["a"][:args.DATASETS]:
        name = os.path.join(data_path, f"pythia.Z_65_155.taupol.rhorho.1M.{kind}.outTUPLE_labFrame")
        print(f"  ==> {letter}, {name}")
        data, weights = read_raw_asci(name, num_particles=7)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    
    return all_data, all_weights


def prepare_z(args):
    data_path = os.path.normpath(args.IN)
    data_copy = []
    n_events = 0

    for i in range(0, 21, 2):
        if i == 0:
            filename = f"CPmix_0"
        elif i < 10:
            filename = f"CPmix_0{i}"
        else:
            filename = f"CPmix_{i}"
        
        # Loading data and parsing it to data and weights
        data, weights = read_raw_all(args, kind=filename)

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
