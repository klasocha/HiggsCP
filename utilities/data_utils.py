import numpy as np
import os

class Dataset(object):
    """ Represent a dataset with features, weights, and additional attributes.
    Provide methods for shuffling the dataset and retrieving batches of data. """

    def __init__(self, x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s):
        self.x = x[:, :-1]
        self.filt = x[:, -1]
        self.weights = weights
        self.argmaxs = argmaxs
        self.c012s = c012s
        self.hits_argmaxs = hits_argmaxs
        self.hits_c012s = hits_c012s

        self.n = x.shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.weights = self.weights[perm]
        self.argmaxs = self.argmaxs[perm]
        self.c012s = self.c012s[perm]
        self.hits_argmaxs = self.hits_argmaxs[perm]
        self.hits_c012s = self.hits_c012s[perm]
        self.filt = self.filt[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return (self.x[cur_id:cur_id+batch_size],
                self.weights[cur_id:cur_id+batch_size], self.argmaxs[cur_id:cur_id+batch_size], self.c012s[cur_id:cur_id+batch_size],
                self.hits_argmaxs[cur_id:cur_id+batch_size], self.hits_c012s[cur_id:cur_id+batch_size], self.filt[cur_id:cur_id+batch_size])


def read_np(filename):
    """ Return the data loaded from a NPY file """
    with open(filename, 'rb') as f:
        return np.load(f)


class EventDatasets(object):
    def __init__(self, event, weights, argmaxs, perm, c012s, hits_argmaxs, 
                 hits_c012s, args, filtered=False, raw=False, miniset=False):
        data = event.cols[:, :-1]
        filt = event.cols[:, -1]

        # Split points for the training, validation, test data set
        data_len = len(data)
        split_1 = round(data_len * 0.1)
        split_2 = round(data_len * 0.2)
        # split_3 = round(data_len * 0.3)
        
        if miniset:
            print("The mini version of the training data set will be used.")
            train_ids = perm[-split_3:-split_2]
        else:
            train_ids = perm[:-split_2]

        valid_ids = perm[-split_2:]
        # valid_ids = perm[-split_2:-split_1]
        # test_ids = perm[-split_1:]

        if not raw:
            print("Data (input features) will be standardised.")
            means = data[train_ids].mean(0)
            stds = data[train_ids].std(0)
            data = (data - means) / stds
            # Saving std and mean
            std_and_means = [stds, means]
            with open(
                os.path.join(
                    args.IN, f"training_std_and_mean_{args.FEAT}_{args.NUM_CLASSES}.npy"), 
                    'wb') as f:
                np.save(f, std_and_means)

        if filtered:
            train_ids = train_ids[filt[train_ids] == 1]
            valid_ids = valid_ids[filt[valid_ids] == 1]
            # test_ids = test_ids[filt[test_ids] == 1]

        data = np.concatenate([data, filt.reshape([-1, 1])], 1)

        self.train = Dataset(data[train_ids], weights[train_ids, :], argmaxs[train_ids], c012s[train_ids], 
                             hits_argmaxs[train_ids], hits_c012s[train_ids])
        
        self.valid = Dataset(data[valid_ids], weights[valid_ids, :], argmaxs[valid_ids], c012s[valid_ids], 
                             hits_argmaxs[valid_ids], hits_c012s[valid_ids])
        
        # self.test = Dataset(data[test_ids], weights[test_ids, :], argmaxs[test_ids], c012s[test_ids], 
        #                     hits_argmaxs[test_ids], hits_c012s[test_ids])