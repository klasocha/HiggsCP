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
argmaxs[0:5]:
{data.train.argmaxs[0:5]}

weights.shape:
{data.train.weights.shape}
np.argmax(weights[0:5], axis=1):
{np.argmax(data.train.weights[0:5], axis=1)}

hits_argmaxs.shape:
{data.train.hits_argmaxs.shape}
np.argmax(hits_argmaxs[0:5], axis=1):
{np.argmax(data.train.hits_argmaxs[0:5], axis=1)}
    """)