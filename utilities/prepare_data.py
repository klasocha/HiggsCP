""" This module is reponsible for preparing data: downloading, preprocessing,
creating ready event objects, unweighted records etc. """
import os
from .cpmix_utils import preprocess_data
from .download_data_rhorho import download_data
from .rhorho import RhoRhoEvent
from .data_utils import EventDatasets
import pickle


def prepare_data(args):
    # Getting the command-line arguments
    num_classes = args.NUM_CLASSES

    # ==================================== DATA PREPARATION ============================================
    if args.EXP != "Z":
        print("\033[1mDownloading data...\033[0m")
        download_data(args)
    
    print("\033[1mPreprocessing data...\033[0m")
    data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s = preprocess_data(args)

    # Saving the RhoRhoEvent object as a pickle binary file for the later analysis 
    # of its attributes (e.g. drawing the distribution of the phistar depending on y1 and y2)
    event = RhoRhoEvent(data, args)
    event_path = os.path.join(args.IN, f"rhorho_event_{args.FEAT}.obj")
    with open(event_path, 'wb') as f:
        pickle.dump(event, f)
    print(f"RhoRhoEvent object has been saved in {event_path}")

    # Saving additionally the EventDatasets object as a pickle binary file
    points = EventDatasets(event, weights, argmaxs, perm, c012s=c012s, hits_argmaxs=hits_argmaxs,  
                           hits_c012s=hits_c012s, args=args, miniset=args.MINISET)
    points_path = os.path.join(args.IN, 
                               f"event_datasets_{num_classes}_{args.HITS_C012s}_{args.FEAT}.obj")
    with open(points_path, 'wb') as f:
        pickle.dump(points, f)
    print(f"EventDatasets object has been saved in {points_path}")

    num_features = points.train.x.shape[1]
    print(f"{num_features} features have been prepared.")