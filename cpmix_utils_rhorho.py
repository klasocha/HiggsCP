import numpy as np
import os
from data_utils import read_np
from scipy import optimize


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)

# here weights and arg_maxs are calculated from continuum distributions
def calc_weights_and_arg_maxs(classes, popts, data_len, num_classes):
    arg_maxs = np.zeros(data_len)
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(classes, *popts[i])
        arg_max = 0
        if weight_fun(2 * np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
            arg_max = 2 * np.pi
        phi = np.arctan(popts[i][2] / popts[i][1])

        if 0 < phi < 2 * np.pi and weight_fun(phi, *popts[i]) > weight_fun(arg_max, *popts[i]):
            arg_max = phi
        if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *popts[i]) > weight_fun(arg_max, *popts[i]):
            arg_max = phi + np.pi
        if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *popts[i]) > weight_fun(arg_max,
                                                                                                   *popts[i]):
            arg_max = phi + 2 * np.pi

        arg_maxs[i] = arg_max
    return weights, arg_maxs


def preprocess_data(args):
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    reuse_weigths = args.REUSE_WEIGTHS  # Set this flag to true if you want reuse calculated weights

    print "Loading data"
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w = read_np(os.path.join(data_path, "rhorho_raw.w.npy")).swapaxes(0, 1)
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print "Read %d events" % data.shape[0]

    data_len = data.shape[0]
    classes = np.linspace(0, 2, num_classes) * np.pi

    if not os.path.exists(os.path.join(data_path, 'popts.npy')):
        popts = np.zeros((data_len, 3))
        pcovs = np.zeros((data_len, 3, 3))
        # here x correspond to values of CPmix at thich data were generated
        x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
        for i in range(data_len):
            popt, pcov = optimize.curve_fit(weight_fun, x, w[i, :], p0=[1, 1, 1])
            popts[i] = popt
            pcovs[i] = pcov

        np.save(os.path.join(data_path, 'popts.npy'), popts)
        np.save(os.path.join(data_path, 'pcovs.npy'), pcovs)
    popts = np.load(os.path.join(data_path, 'popts.npy'))

    if not reuse_weigths or not os.path.exists(os.path.join(data_path, 'weigths.npy')) \
            or not os.path.exists(os.path.join(data_path, 'arg_maxs.npy')) \
            or np.load(os.path.join(data_path, 'weigths.npy')).shape[1] != num_classes:
        weights, arg_maxs = calc_weights_and_arg_maxs(classes, popts, data_len, num_classes)
        np.save(os.path.join(data_path, 'weigths.npy'), weights)
        np.save(os.path.join(data_path, 'arg_maxs.npy'), arg_maxs)
    weights  = np.load(os.path.join(data_path, 'weigths.npy'))
    arg_maxs = np.load(os.path.join(data_path, 'arg_maxs.npy'))

    #ERW
    # here arg_maxs are in fraction of pi, not in the class index
    # how we go then from fraction of pi to class index??
    # print "preprocess: weights", weights
    # print "preprocess: arg_maxs", arg_maxs

    #ERW
    # I am not sure what the purpose is and if it make sens.
    if args.RESTRICT_MOST_PROBABLE_ANGLE:
        arg_maxs[arg_maxs > np.pi] = -1 * arg_maxs[arg_maxs > np.pi] + 2 * np.pi

    #ERW
    # this optimisation does not help, revisit, maybe not correctly implemented?
    if args.NORMALIZE_WEIGHTS:
        weights = weights/np.reshape(popts[:, 0], (-1, 1))
        
    # ERW    
    # here weights and arg_maxs are calculated at value of CPmix representing given class
    # in training, class is expressed as integer, not fraction pf pi.
    return data, weights, arg_maxs, perm, popts
