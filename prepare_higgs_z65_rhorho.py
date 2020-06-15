import numpy as np
from prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(kind, args):
    print "Reading %s" % kind

    data_path = args.IN

    all_data = []
    all_weights = []

    if kind == "Higgs":
        for letter in ["a", "b"][:args.HDATASETS]:
            name = os.path.join(data_path, "pythia.H.rhorho.1M.%s.scalar.outTUPLE_labFrame" % (letter))
            print letter, name
            data, weights = read_raw_root(name, num_particles=7)
            all_data += [data]
            all_weights += [weights]
        all_data = np.concatenate(all_data)
        all_weights = np.concatenate(all_weights)
        return all_data, all_weights, np.zeros_like(all_weights)
    else:
        for letter in ["c", "d"][:args.ZDATASETS]:
            name = os.path.join(data_path, "pythia.Z_65_155.rhorho.1M.%s.outTUPLE_labFrame" % (letter))
            print letter, name
            data, weights = read_raw_root(name, num_particles=7)
            all_data += [data]
            all_weights += [weights]
        all_data = np.concatenate(all_data)
        all_weights = np.concatenate(all_weights)
        return all_data, np.zeros_like(all_weights), all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-z", "--Zdatasets", dest="ZDATASETS", default='1', type=int, help="number of Z datasets to prepare")
    parser.add_argument("-higgs", "--Hdatasets", dest="HDATASETS", default='1', type=int, help="number of H datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["RHORHO_DATA_org"])
    args = parser.parse_args()

    data_H, weights_aH, weights_bH = read_raw_all("Higgs", args)
    data_Z, weights_aZ, weights_bZ  = read_raw_all("Z", args)

    data = np.vstack((data_H, data_Z))
    weights_a = np.append(weights_aH, weights_aZ)
    weights_b = np.append(weights_bH, weights_bZ)

    np.random.seed(123)
    perm = np.random.permutation(len(weights_a))

    data_path = args.IN

    np.save(os.path.join(data_path, "rhorho_Higgs_Z_raw.data.npy"), data)
    np.save(os.path.join(data_path, "rhorho_Higgs_Z_raw.w_a.npy"), weights_a)
    np.save(os.path.join(data_path, "rhorho_Higgs_Z_raw.w_b.npy"), weights_b)
    np.save(os.path.join(data_path, "rhorho_Higgs_Z_raw.perm.npy"), perm)
