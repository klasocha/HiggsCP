""" This program read the dowloaded original raw data, parse all the records and then
saved them as prepared "rhorho_raw.*.npy: files """
import numpy as np
from .prepare_utils import read_raw_asci
import os, config


def read_raw_all(args, kind=None, filename=None):
    """ Read the original raw data and use src_py.prepare_utils.read_raw_asci to parse it
    as data and weights ready for being saved as "rhorho_raw.*.npy files". """
    print(f"Reading and parsing the raw data from {kind if filename is None else filename}")

    data_path = args.IN
    all_data = []
    all_weights = []

    if args.DATA_FORMAT == "v1":
        for letter in ["a"][:args.DATASETS]:
            name = os.path.join(data_path, "pythia.H.rhorho.1M.%s.%s.outTUPLE_labFrame" % (letter, kind))
            print(f"  ==> {letter}, {name}")
            # one header (TUPLE), six momenta
            data, weights = read_raw_asci(name, num_particles=1+6)
            all_data += [data]
            all_weights += [weights]
    
    if args.DATA_FORMAT == "v2":
        name = os.path.join(data_path, filename)
        # one header (TUPLE), nine lines containing vectors/numbers
        data, weights = read_raw_asci(name, num_particles=1+9)
        all_data += [data]
        all_weights += [weights]

    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    
    return all_data, all_weights


def prepare_rhorho(args):
    data_path = os.path.normpath(args.IN)
    data_copy = []
    n_events = 0

    if args.DATA_FORMAT == "v1":
        for i in range(0, 21):
            if i < 10:
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
            with open(os.path.join(data_path, weights_path), "wb") as f:
                np.save(f, weights)

        # Saving data
        with open(os.path.join(data_path, "rhorho_raw.data.npy"), "wb") as f:
            np.save(f, data_copy)

    if args.DATA_FORMAT == "v2":
        for i, filename in zip(range(0, len(config.DATA_RUN_2_FILES)), config.DATA_RUN_2_FILES):
            # Loading data and parsing it to data and weights
            data, independent_weights = read_raw_all(args, kind="events formatted as Run 2 data", 
                                                     filename=filename)
            
            # Picking only "rho-rho" events
            mask_rho1 = data[:, 4] == 1
            mask_rho2 = data[:, 17] == 1
            mask_rhorho = mask_rho1 * mask_rho2
            data_copy = data[mask_rhorho]
            independent_weights = independent_weights[mask_rhorho]
            print(f"Found {len(data_copy)} rho-rho events.")
            # Removing "rho-rho" identifiers
            data_copy = np.delete(data_copy, [4, 17], axis=1)
            
            # Saving weights which do not depend on alphaCP
            n_events += len(independent_weights)
            with open(os.path.join(data_path, f"rhorho_raw.w_independent_{i}.npy"), "wb") as f:
                np.save(f, independent_weights)
            
            # Saving alphaCP-dependent weights (phiCP 18 hypotheses: 0°-180°)
            with open(os.path.join(data_path, f"rhorho_raw.w_{i}.npy"), "wb") as f:
                np.save(f, data_copy[:, 25:-1])
            
            # Removing alphaCP-dependent weights from the data
            data_copy = np.delete(data_copy, [col for col in range(25, 25 + 18)], axis=1)

            # Saving data
            with open(os.path.join(data_path, f"rhorho_raw.data_{i}.npy"), "wb") as f:
                np.save(f, data_copy)

    # Preparing permutations for data shuffling
    prepare_permutations(n_events, data_path)
    print(f"In total: prepared {n_events} events.")


def prepare_permutations(n_events, data_path):
    np.random.seed(123)
    perm = np.random.permutation(n_events)
    with open(os.path.join(data_path, "rhorho_raw.perm.npy"), "wb") as f:
        np.save(f, perm)
    return perm