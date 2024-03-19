import numpy as np
import os
from scipy import optimize
from .data_utils import read_np


def weight_fun(alphaCP, c0, c1, c2):
    """ Calculate the spin weight by using the alphaCP angle [0, 2*pi] 
    corresponding the scalar state, as well as the C0/C1/C2 coefficients """
    return c0 + c1 * np.cos(alphaCP) + c2 * np.sin(alphaCP)


def hits_fun(classes, x, num_classes):
    """ Assign x to one of the intervals defined by the classes. 
    Return a vector which is the one-hot encoded representation of x assigned
    to a specific bin among all the len(classes) bins available """
    hits = np.zeros(num_classes)

    for i in range(num_classes - 1):
        if classes[i] <= x < classes[i + 1]:
          hits[i] = 1.0
    if x >= classes[num_classes - 1]:
        hits[i] = 1.0
       
    return hits


def calc_hits_c012s(classes, c012s, data_len, num_classes):
    """ Prepares data for learning the coefficients C0, C1, C2 and drawing 
    their distribution plots (hits maps are calculated) """
    hits_c0s = np.zeros((data_len, num_classes))
    hits_c1s = np.zeros((data_len, num_classes))
    hits_c2s = np.zeros((data_len, num_classes))

    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been processed...", end='\r')
        hits_c0s[i] = hits_fun(classes, c012s[i][0], num_classes)
        hits_c1s[i] = hits_fun(classes, c012s[i][1] + 1.0, num_classes)
        hits_c2s[i] = hits_fun(classes, c012s[i][2] + 1.0, num_classes)
    print()
    return hits_c0s, hits_c1s, hits_c2s


def calc_weights_and_argmaxs(classes, c012s, data_len, num_classes):
    """ Calculate weights and argmax values from continuum distributions. """
    argmaxs     = np.zeros((data_len, 1))
    weights      = np.zeros((data_len, num_classes))
    hits_argmaxs = np.zeros((data_len, num_classes))

    print("Calculating weights and argmax values from continuum distributions")    
    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been processed...", end='\r')
        weights[i] = weight_fun(classes, *c012s[i])
        arg_max = 0
        if weight_fun(2 * np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = 2 * np.pi
        phi = np.arctan(c012s[i][2] / c012s[i][1])

        if 0 < phi < 2 * np.pi and weight_fun(phi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi

        if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi + np.pi
        
        if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi + 2 * np.pi

        argmaxs[i] = arg_max
        hits_argmaxs[i] = hits_fun(classes, arg_max, num_classes)
    print()
    return weights, argmaxs, hits_argmaxs


def preprocess_data(args):
    """ Preprocesses the data for training, including loading, calculating coefficients, 
    and transforming data into suitable formats. """
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    # Set this flag to true if you want reuse calculated weights:
    reuse_weights = args.REUSE_WEIGHTS  

    # Extracting the decay mode (e.g. "nn_rhorho" -> "rhorho")
    suffix = (args.TYPE).split("_")[-1] #-1 to indeks ostatniego elementu 
    
    # Reading the data
    print("Loading raw data")
    data = read_np(os.path.join(data_path, suffix + "_raw.data.npy"))
    w = read_np(os.path.join(data_path, suffix + "_raw.w.npy")).swapaxes(0, 1)
    perm = read_np(os.path.join(data_path, suffix + "_raw.perm.npy"))
    print(f"Read {data.shape[0]} events")

    data_len = data.shape[0]

    # Calculating and saving the C coefficients
    if args.FORCE_DOWNLOAD or not os.path.exists(os.path.join(data_path, 'c012s.npy')):
        # Array to store C0, C1, and C2 coefficients (per event) 
        # It will be the input for the regression or softmax
        c012s   = np.zeros((data_len, 3))

        # Array to store covariance matrices per data point  
        ccovs  = np.zeros((data_len, 3, 3))
        
        # Values of CPmix at which data were generated
        x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
        print("Calculating C0/C1/C2 and the covariance with scipy.optimize.curve_fit()")
        for i in range(data_len):
            if i % 10000 == 0:
                print(f"{i} events have been used by scipy.optimize.curve_fit()", end='\r')
            coeff, ccov = optimize.curve_fit(weight_fun, x, w[i, :], p0=[1, 1, 1])
            c012s[i]  = coeff
            ccovs[i]  = ccov

        # Saving the coefficients and covariance as NPY files
        np.save(os.path.join(data_path, 'c012s.npy'), c012s)
        np.save(os.path.join(data_path, 'ccovs.npy'), ccovs)
    
    # Loading C0, C1, C2 and saving them in a one-hot encoded form
    c012s = read_np(os.path.join(data_path, 'c012s.npy'))
    c012s_paths = []
    for i in range(3):
        c012s_paths.append(os.path.join(data_path, f'hits_c{i}s.npy'))

    if args.FORCE_DOWNLOAD or not (os.path.exists(c012s_paths[0]) \
        and os.path.exists(c012s_paths[1]) \
        and os.path.exists(c012s_paths[2]) \
        and read_np(c012s_paths[0]).shape[1] == num_classes \
        and read_np(c012s_paths[1]).shape[1] == num_classes \
        and read_np(c012s_paths[2]).shape[1] == num_classes):
        classes = np.linspace(0, 2, num_classes) 
        print("Converting the C0/C1/C1 coefficients to a one-hot encoded format") 
        hits_c0s, hits_c1s, hits_c2s = calc_hits_c012s(classes, c012s, data_len, num_classes)
        print("Saving the C0/C1/C2 coefficients in one-hot encoded form")
        np.save(c012s_paths[0], hits_c0s)
        np.save(c012s_paths[1], hits_c1s)
        np.save(c012s_paths[2], hits_c2s)

    if args.HITS_C012s == "hits_c0s" :
        hits_c012s = read_np(c012s_paths[0])
    elif args.HITS_C012s == "hits_c1s" :   
        hits_c012s = read_np(c012s_paths[1])
    elif args.HITS_C012s == "hits_c2s" :   
        hits_c012s = read_np(c012s_paths[2])

    # Calculating the weights and argmaxes (one-hot encoded) and saving them
    weights_path = os.path.join(data_path, f'weights_multiclass_{num_classes}.npy')
    argmaxs_path = os.path.join(data_path, 'argmaxs.npy')
    hits_argmaxs_path = os.path.join(data_path, 'hits_argmaxs.npy')

    if args.FORCE_DOWNLOAD or not (reuse_weights and os.path.exists(weights_path) \
        and os.path.exists(argmaxs_path) \
        and os.path.exists(hits_argmaxs_path) \
        and read_np(weights_path).shape[1] == num_classes \
        and read_np(hits_argmaxs_path).shape[1] == num_classes):
        classes = np.linspace(0, 2, num_classes) * np.pi
        weights, argmaxs, hits_argmaxs = calc_weights_and_argmaxs(classes, c012s, data_len, num_classes)

        np.save(weights_path, weights)
        np.save(argmaxs_path, argmaxs)
        np.save(hits_argmaxs_path, hits_argmaxs)

    weights  = read_np(weights_path)
    argmaxs = read_np(argmaxs_path)
    hits_argmaxs = read_np(hits_argmaxs_path)

    # Unweighting the events and saving the "hits"
    unweighted_events_weights_filename = f"unwt_multiclass_{num_classes}.npy"
    weights_normalised = weights / 2
    data_len = len(weights_normalised)
    unweighted_events = []
    monte_carlo = lambda x : 0.0 if x < np.random.random() else 1.0

    print(f"Unweighting the events...", end='\r')
    unweighted_events = np.vectorize(monte_carlo)(weights_normalised)
    
    output_path = os.path.join(data_path, unweighted_events_weights_filename)
    np.save(output_path, unweighted_events)
    print(f"Weights of the unweighted events have been saved in {output_path}")

    # TODO: Revisit
    # Comment from ERW:
    # Here, argmax values are represented as fractions of pi, not as class indices.
    # We need to determine how to convert from fractions of pi to class indices.
    # Uncomment the following lines to print the weights and argmax values for preprocessing:
    # print("Preprocessing: weights", weights)
    # print("Preprocessing: argmaxs", argmaxs)

    # TODO: Revisit.
    # Comment from ERW:
    # I am not sure of the purpose of this code, and whether it makes sense.
    if args.RESTRICT_MOST_PROBABLE_ANGLE:
        argmaxs[argmaxs > np.pi] = -1 * argmaxs[argmaxs > np.pi] + 2 * np.pi

    # Comment from ERW:
    # This optimization process does not provide the expected improvement. 
    # TODO: Revisit the implementation; it's possible that it has not been correctly implemented.
    if args.NORMALIZE_WEIGHTS:
        weights = weights / np.reshape(c012s[:, 0], (-1, 1))
        
    # Comment from ERW:
    # Here, weights and argmax values are calculated at the value of CPmix representing a given class.
    # In training, the class is expressed as an integer, not as a fraction of pi.
    return data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s
