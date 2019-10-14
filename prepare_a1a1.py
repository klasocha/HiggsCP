import numpy as np
from prepare_utils import read_raw_root
import argparse
import os


def read_raw_all(kind, args, letter):
    print "Reading %s" % kind

    data_path = args.IN

    
    all_data = []
    all_weights = []
    for number in ["1", "2", "3", "4"]:
        name = os.path.join(data_path, "pythia.H.a1a1.1M.corr.%s_%s.CPmix_%s.outTUPLE_labFrame" % (letter, number, kind))
        print letter, number, name
        data, weights = read_raw_root(name, num_particles=9)
        all_data += [data]
        all_weights += [weights]
    all_data = np.concatenate(all_data)
    all_weights = np.concatenate(all_weights)
    return all_data, all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument("-d", "--datasets", dest="DATASETS", default='10', type=int, help="number of datasets to prepare")
    parser.add_argument("-i", "--input", dest="IN", default=os.environ["A1A1_DATA"])
    args = parser.parse_args()
    
    '''for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"][:args.DATASETS]:
        data_a, weights_a = read_raw_all("scalar", args, letter)
        data_b, weights_b = read_raw_all("pseudoscalar", args, letter)

        print "In total: %d examples." % len(weights_a)

        part = len(weights_a) // 10
        for i in range(10):
            np.testing.assert_array_almost_equal(data_a[i*part: i*part + part], data_b[i*part: i*part + part])
        np.testing.assert_array_almost_equal(data_a[10*part:], data_b[10*part:])    


        #np.random.seed(123)
        #perm = np.random.permutation(len(weights_a))

        data_path = args.IN

        np.save(os.path.join(data_path, "a1a1_raw.data%s.npy" %(letter)), data_a)
        np.save(os.path.join(data_path, "a1a1_raw.w_a%s.npy" %(letter)), weights_a)
        np.save(os.path.join(data_path, "a1a1_raw.w_b%s.npy" %(letter)), weights_b)    

    
    data_path = args.IN
    data_a = np.load(os.path.join(data_path, "a1a1_raw.dataa.npy"))
    #data_a = []
    weights_a = []
    weights_b = []
    
    for letter in [ "b", "c", "d", "e", "f", "g", "h", "i", "k"][:args.DATASETS]:   
        print(np.shape(np.load(os.path.join(data_path, "a1a1_raw.data%s.npy" %(letter)))))
        data_a = np.vstack((data_a, np.load(os.path.join(data_path, "a1a1_raw.data%s.npy" %(letter)))))
        #data_a = np.append(data_a, np.load(os.path.join(data_path, "a1a1_raw.data%s.npy" %(letter))))
    np.save(os.path.join(data_path, "a1a1_raw.data.npy"), data_a)
    del data_a'''

    data_path = args.IN
    weights_a = []
    weights_b = []

    
    for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"][:args.DATASETS]:
        weights_a = np.append(weights_a, np.load(os.path.join(data_path, "a1a1_raw.w_a%s.npy" %(letter))))
        weights_b = np.append(weights_b, np.load(os.path.join(data_path, "a1a1_raw.w_b%s.npy" %(letter))))
    

    np.random.seed(123)
    perm = np.random.permutation(len(weights_a))
    np.save(os.path.join(data_path, "a1a1_raw.w_a.npy"), weights_a)
    np.save(os.path.join(data_path, "a1a1_raw.w_b.npy"), weights_b)
    np.save(os.path.join(data_path, "a1a1_raw.perm.npy"), perm)
    
