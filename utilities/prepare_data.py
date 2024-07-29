""" This module is reponsible for preparing data: downloading, preprocessing,
creating ready event objects, unweighted records etc. """
import os, pickle, numpy as np
from .cpmix_utils import preprocess_data
from .download_data_rhorho import download_data
from .rhorho import RhoRhoEvent
from .data_utils import EventDatasets
import config


def prepare_data(args, preprocess_only=False):
    # Getting the command-line arguments
    num_classes = args.NUM_CLASSES

    # ============================ DATA PREPARATION ============================
    if args.EXP == "Z":
        download_data(args)
    elif not preprocess_only:
        print("\033[1mDownloading data...\033[0m")
        download_data(args)
    
    print("\033[1mPreprocessing data...\033[0m")
    if args.DATA_FORMAT == "v1":
        if preprocess_only:
            weights = []
            CPmix_index = [
                "00", # scalar
                "02", "04", "06", "08", 
                "10", # pseudoscalar
                "12", "14", "16", "18", 
                "20"  # scalar
            ] 
            
            for index in CPmix_index:
                filename = 'rhorho_raw.w_' + index + '.npy'
                filepath = os.path.join(args.IN, filename)
                with open(filepath, "rb") as f:
                    weights.append(np.load(f))
        
            # Joining and then saving all the parts together in a single file
            weights = np.stack(weights)
            all_weights_output_path = os.path.join(args.IN, "rhorho_raw.w.npy")
            with open(all_weights_output_path, "wb") as f:
                np.save(f, weights)

    if args.DATA_FORMAT == "v2":
        if preprocess_only:
            indices = range(0, len(config.DATA_RUN_2_FILES))
            kinds = ["w", "data", "w_independent"]

            for kind in kinds:
                if not os.path.exists(os.path.join(args.IN, f"rhorho_raw.{kind}.npy")):
                    objects = None
                    for i in indices:
                        filename = f"rhorho_raw.{kind}_{i}.npy"
                        with open(os.path.join(args.IN, filename), "rb") as f:
                            if i == 0:
                                objects = np.load(f)
                            else:
                                objects = np.append(objects, np.load(f), axis=0)
                    with open(os.path.join(args.IN, f"rhorho_raw.{kind}.npy"), "wb") as f:
                        np.save(f, objects)
                
                    if os.path.exists(os.path.join(args.IN, f"rhorho_raw.{kind}.npy")):
                        for i in indices:
                            os.remove(os.path.normpath(os.path.join(args.IN, f"rhorho_raw.{kind}_{i}.npy")))

    data, weights, argmaxs, perm, c012s, hits_argmaxs, \
        hits_c012s = preprocess_data(args)

    # Saving the RhoRhoEvent object as a pickle binary file for the later 
    # analysis of its attributes (e.g. drawing the distribution of the phistar 
    # depending on y1 and y2)
    event = RhoRhoEvent(data, args)
    event_path = os.path.join(args.IN, f"rhorho_event_{args.FEAT}.obj")
    with open(event_path, 'wb') as f:
        pickle.dump(event, f)
    print(f"RhoRhoEvent object has been saved in {event_path}")

    # Saving additionally the EventDatasets object as a pickle binary file
    points = EventDatasets(event, weights, argmaxs, perm, c012s=c012s, 
                           hits_argmaxs=hits_argmaxs, hits_c012s=hits_c012s, 
                           args=args, miniset=args.MINISET)
    points_path = os.path.join(
        args.IN, 
        f"event_datasets_{num_classes}_{args.HITS_C012s}_{args.FEAT}.obj")
    
    with open(points_path, 'wb') as f:
        pickle.dump(points, f)
    print(f"EventDatasets object has been saved in {points_path}")

    num_features = points.train.x.shape[1]
    print(f"{num_features} features have been prepared.")