import numpy as np
import pickle, os

def test_labels(args):
    data_points_path = os.path.join(
        args.IN, 
        f"event_datasets_{args.NUM_CLASSES}_{args.HITS_C012s}_{args.FEAT}.obj")
    with open(data_points_path, 'rb') as f:
        data = pickle.load(f)

    print(f"""
Example for training data and 51 classes:
    
argmaxs.shape:
{data.train.argmaxs.shape}
argmaxs[0:10]:
{data.train.argmaxs[0:10]}

weights.shape:
{data.train.weights.shape}
np.argmax(weights[0:10], axis=1):
{np.argmax(data.train.weights[0:10], axis=1)}

hits_argmaxs.shape:
{data.train.hits_argmaxs.shape}
np.argmax(hits_argmaxs[0:10], axis=1):
{np.argmax(data.train.hits_argmaxs[0:10], axis=1)}


hits_c012s.shape:
{data.train.hits_c012s.shape}
hits_c012[0:10] => c012s[0:10]:
{np.argmax(data.train.hits_c012s[0:10], axis=1) / (int(args.NUM_CLASSES) - 1) * 2}

c012s.shape:
{data.train.c012s.shape}
c012s[0:10]:
{data.train.c012s[0:10]}


np.argmax(weights) == np.argmax(hits_argmax):
{np.sum(np.argmax(data.train.weights, axis=1) ==
  np.argmax(data.train.hits_argmaxs, axis=1))}
    """)