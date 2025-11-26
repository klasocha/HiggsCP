import numpy as np
import os
from scipy import optimize
from .data_utils import read_np
from .prepare_rhorho import prepare_permutations


def weight_fun(alphaCP, c0, c1, c2):
    """ Calculate the spin weight by using the alphaCP angle [0, 2*pi] 
    corresponding the scalar state, as well as the C0/C1/C2 coefficients """
    return c0 + c1 * np.cos(alphaCP) + c2 * np.sin(alphaCP)


def hits_fun(classes, x, num_classes, periodicity=False):
    """ Assign x to one of the intervals defined by the classes. 
    Return a vector which is the one-hot encoded representation of x assigned
    to a specific bin among all the len(classes) bins available. If
    periodicity is True, then it does not convert values to one-hot encoded ones
    as the first bin and the last bin are assigned to its common sum. """

    hits = np.zeros(num_classes)

    if x < ((classes[0] + classes[1]) / 2):
        hits[0] = 1.0
        
    for i in range(1, num_classes):
        if ((classes[i-1] + classes[i]) / 2) <= x < \
            ((classes[i] + classes[i+1]) / 2):
            hits[i] = 1.0
    
    if periodicity:
        if hits[0] == 1:
            hits[num_classes - 1] = 1
        if hits[num_classes - 1] == 1:
            hits[0] = 1   

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
    if data_len < 10000:
        print(f"{data_len} events have been processed...", end='\r')
    print()
    return hits_c0s, hits_c1s, hits_c2s


def calc_weights_and_argmaxs(c012s, data_len, num_classes):
    """ Calculate weights and argmax values from continuum distributions. """
    argmaxs     = np.zeros((data_len, 1))
    weights      = np.zeros((data_len, num_classes))
    hits_argmaxs = np.zeros((data_len, num_classes))
    
    classes_for_weight_fun = np.linspace(0, 2, num_classes) * np.pi
    classes_for_hits_fun = np.linspace(0, 2 + 2/(num_classes - 1), (num_classes + 1)) * np.pi

    print("Calculating weights and argmax values from continuum distributions")    
    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been processed...", end='\r')
        weights[i] = weight_fun(classes_for_weight_fun, *c012s[i])
        arg_max = 0

        phi = np.arctan(c012s[i][2] / c012s[i][1])

        if 0 < phi < 2 * np.pi and weight_fun(phi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi

        if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi + np.pi
        
        if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi + 2 * np.pi

        argmaxs[i] = arg_max
        hits_argmaxs[i] = hits_fun(classes_for_hits_fun, arg_max, num_classes, True)
    if data_len < 10000:
        print(f"{data_len} events have been processed...", end='\r')
    print()
    return weights, argmaxs, hits_argmaxs


def preprocess_data(args):
    """ Preprocesses the data for training, including loading, calculating coefficients, 
    and transforming data into suitable formats. """
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    # Set this flag to true if you want reuse calculated weights:
    reuse_weights = args.REUSE_WEIGHTS  

    # Specifying the decay mode
    suffix = "rhorho"
    
    # Reading the data
    print("Loading raw data")
    data = read_np(os.path.join(data_path, suffix + "_raw.data.npy"))
    if args.DATA_FORMAT == "v1":
        w = read_np(os.path.join(data_path, suffix + "_raw.w.npy")).swapaxes(0, 1)
        perm = read_np(os.path.join(data_path, suffix + "_raw.perm.npy"))
    if args.DATA_FORMAT == "v2":
        w = read_np(os.path.join(data_path, suffix + "_raw.w.npy"))
        perm = read_np(os.path.join(data_path, suffix + "_raw.perm.npy"))
    if args.DATA_FORMAT in ["v3", "v4"]:
        w = read_np(os.path.join(data_path, suffix + "_raw.w.npy"))
        perm = prepare_permutations(n_events=w.shape[0], data_path=data_path)
        phistar = read_np(os.path.join(data_path, suffix + "_raw.phistar.npy"))
        phistar = np.reshape(phistar, (phistar.shape[0], 1))
        data = np.concatenate([data, phistar], axis=1)
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
        if args.DATA_FORMAT == "v1":
            x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
        if args.DATA_FORMAT in ["v2", "v3", "v4"]:
            # alphaCP = 2 * phiCP (aka Theta)
            x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 
                          2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]) * 100 / 180 * np.pi
        
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
        classes = np.linspace(0, 2 + 2/(num_classes - 1), (num_classes + 1)) 
        print("Converting the C0/C1/C1 coefficients to a one-hot encoded format") 
        hits_c0s, hits_c1s, hits_c2s = calc_hits_c012s(classes, c012s, data_len, num_classes)
        print("Saving the C0/C1/C2 coefficients in one-hot encoded form")
        np.save(c012s_paths[0], hits_c0s)
        np.save(c012s_paths[1], hits_c1s)
        np.save(c012s_paths[2], hits_c2s)

    if args.HITS_C012s == "hits_c0s" :
        print(f"Choosing hits_c0s")
        hits_c012s = read_np(c012s_paths[0])
    elif args.HITS_C012s == "hits_c1s" :   
        print(f"Choosing hits_c1s")
        hits_c012s = read_np(c012s_paths[1])
    elif args.HITS_C012s == "hits_c2s" :   
        print(f"Choosing hits_c2s")
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
        weights, argmaxs, hits_argmaxs = calc_weights_and_argmaxs(c012s, data_len, num_classes)

        np.save(weights_path, weights)
        np.save(argmaxs_path, argmaxs)
        np.save(hits_argmaxs_path, hits_argmaxs)

    weights  = read_np(weights_path)
    argmaxs = read_np(argmaxs_path)
    hits_argmaxs = read_np(hits_argmaxs_path)

    if args.EXP != "Z":
        # Unweighting the events and saving the "hits"
        unweighted_events_weights_filename = f"unwt_multiclass_{num_classes}.npy"
        output_path = os.path.join(data_path, unweighted_events_weights_filename)
        if args.FORCE_DOWNLOAD or not (reuse_weights and os.path.exists(output_path) \
            and read_np(output_path).shape[1] == num_classes):
            weights_normalised = weights / 2
            data_len = len(weights_normalised)
            unweighted_events = []
            monte_carlo = lambda x : 0.0 if x < np.random.random() else 1.0
            print(f"Unweighting the events...", end='\r')
            unweighted_events = np.vectorize(monte_carlo)(weights_normalised)
            
            with open(output_path, "wb") as f:
                np.save(f, unweighted_events)
            print(f"Weights of the unweighted events have been saved in {output_path}")

    # Comment from ERW:
    # Here, weights and argmax values are calculated at the value of CPmix representing a given class.
    # In training, the class is expressed as an integer, not as a fraction of pi.
    return data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s
