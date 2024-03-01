import numpy as np
import os
from data_utils import read_np
from scipy import optimize


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

def hits_fun(classes, x, num_classes):

    hits = np.zeros(num_classes)
    for i in range(num_classes-1):
        if x >= classes[i] and  x < classes[i+1]:
             hits[i] = 1.0
#   fix for filling the last bin (?)             
    if( x >=  classes[i+1]):
       hits[i] = 1.0
       
    return hits


# here hits maps are calculated 
def calc_hits_c012s(classes, c012s, data_len, num_classes):
    hits_c0s = np.zeros((data_len, num_classes))
    hits_c1s = np.zeros((data_len, num_classes))
    hits_c2s = np.zeros((data_len, num_classes))
    for i in range(data_len):
        hits_c0s[i] = hits_fun(classes, c012s[i][0], num_classes)
        hits_c1s[i] = hits_fun(classes, c012s[i][1]+1.0, num_classes)
        hits_c2s[i] = hits_fun(classes, c012s[i][2]+1.0, num_classes)

    return hits_c0s, hits_c1s, hits_c2s

# here weights and argmaxs are calculated from continuum distributions
def calc_weights_and_argmaxs(classes, c012s, data_len, num_classes):
    argmaxs     = np.zeros((data_len, 1))
    weights      = np.zeros((data_len, num_classes))
    hits_argmaxs = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(classes, *c012s[i])
        arg_max = 0
        if weight_fun(2 * np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = 2 * np.pi
        phi = np.arctan(c012s[i][2] / c012s[i][1])

        if 0 < phi < 2 * np.pi and weight_fun(phi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi
        if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *c012s[i]) > weight_fun(arg_max, *c012s[i]):
            arg_max = phi + np.pi
        if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *c012s[i]) > weight_fun(arg_max,
                                                                                                   *c012s[i]):
            arg_max = phi + 2 * np.pi

        argmaxs[i] = arg_max
        hits_argmaxs[i] = hits_fun(classes, arg_max, num_classes)

    return weights, argmaxs, hits_argmaxs


def preprocess_data(args):
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    reuse_weights = args.REUSE_WEIGHTS  # Set this flag to true if you want reuse calculated weights

    print "Loading data"
    suffix = (args.TYPE).split("_")[-1] #-1 to indeks ostatniego elementu 
    data = read_np(os.path.join(data_path, suffix + "_raw.data.npy"))
    w = read_np(os.path.join(data_path, suffix + "_raw.w.npy")).swapaxes(0, 1)
    perm = read_np(os.path.join(data_path, suffix + "_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    data_len = data.shape[0]

    if not os.path.exists(os.path.join(data_path, 'c012s.npy')):
        c012s   = np.zeros((data_len, 3))
        ccovs  = np.zeros((data_len, 3, 3))
        # here x correspond to values of CPmix at thich data were generated
        # coeffs is an array for C0, C1, C2 coefficients (per event)
        # being inputs to regression or softmax 
        x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
        for i in range(data_len):
            coeff, ccov = optimize.curve_fit(weight_fun, x, w[i, :], p0=[1, 1, 1])
            c012s[i]  = coeff
            ccovs[i]  = ccov

        np.save(os.path.join(data_path, 'c012s.npy'), c012s)
        np.save(os.path.join(data_path, 'ccovs.npy'), ccovs)

    c012s = np.load(os.path.join(data_path, 'c012s.npy'))
   

    if not os.path.exists(os.path.join(data_path, 'hits_c0s.npy')) \
       or not os.path.exists(os.path.join(data_path, 'hits_c1s.npy')) \
       or not os.path.exists(os.path.join(data_path, 'hits_c2s.npy')) \
       or np.load(os.path.join(data_path, 'hits_c0s.npy')).shape[1] != num_classes \
       or np.load(os.path.join(data_path, 'hits_c1s.npy')).shape[1] != num_classes \
       or np.load(os.path.join(data_path, 'hits_c2s.npy')).shape[1] != num_classes :
        classes = np.linspace(0, 2, num_classes) 
        hits_c0s, hits_c1s, hits_c2s = calc_hits_c012s(classes, c012s, data_len, num_classes)
        np.save(os.path.join(data_path, 'hits_c0s.npy'), hits_c0s)
        np.save(os.path.join(data_path, 'hits_c1s.npy'), hits_c1s)
        np.save(os.path.join(data_path, 'hits_c2s.npy'), hits_c2s)

    if args.HITS_C012s == "hits_c0s" :
        hits_c012s = np.load(os.path.join(data_path, 'hits_c0s.npy'))
    elif args.HITS_C012s == "hits_c1s" :   
        hits_c012s = np.load(os.path.join(data_path, 'hits_c1s.npy'))
    elif args.HITS_C012s == "hits_c2s" :   
        hits_c012s = np.load(os.path.join(data_path, 'hits_c2s.npy'))


    if not reuse_weights or not os.path.exists(os.path.join(data_path, 'weights.npy')) \
            or not os.path.exists(os.path.join(data_path, 'argmaxs.npy')) \
            or not os.path.exists(os.path.join(data_path, 'hits_argmaxs.npy')) \
            or np.load(os.path.join(data_path, 'weights.npy')).shape[1] != num_classes \
            or np.load(os.path.join(data_path, 'hits_argmaxs')).shape[1] != num_classes:
        classes = np.linspace(0, 2, num_classes) * np.pi
        weights, argmaxs,  hits_argmaxs = calc_weights_and_argmaxs(classes, c012s, data_len, num_classes)
        np.save(os.path.join(data_path, 'weights.npy'), weights)
        np.save(os.path.join(data_path, 'argmaxs.npy'), argmaxs)
        np.save(os.path.join(data_path, 'hits_argmaxs.npy'), hits_argmaxs)
    weights  = np.load(os.path.join(data_path, 'weights.npy'))
    argmaxs = np.load(os.path.join(data_path, 'argmaxs.npy'))
    hits_argmaxs = np.load(os.path.join(data_path, 'hits_argmaxs.npy'))

    #ERW
    # here argmaxs are in fraction of pi, not in the class index
    # how we go then from fraction of pi to class index??
    # print "preprocess: weights", weights
    # print "preprocess: argmaxs", argmaxs

    #ERW
    # I am not sure what the purpose is and if it make sens.
    if args.RESTRICT_MOST_PROBABLE_ANGLE:
        argmaxs[argmaxs > np.pi] = -1 * argmaxs[argmaxs > np.pi] + 2 * np.pi

    #ERW
    # this optimisation does not help, revisit, maybe not correctly implemented?
    if args.NORMALIZE_WEIGHTS:
        weights = weights/np.reshape(c012s[:, 0], (-1, 1))
        
    # ERW    
    # here weights and argmaxs are calculated at value of CPmix representing given class
    # in training, class is expressed as integer, not fraction pf pi.

    return data, weights, argmaxs, perm, c012s, hits_argmaxs, hits_c012s
