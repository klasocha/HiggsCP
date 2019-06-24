from __future__ import print_function
import numpy as np
import os
from data_utils import read_np
from scipy import optimize


def weight_fun(x, a, b, c):
    return a + b * np.cos(x) + c * np.sin(x)


def calc_arg_maxs(fitted_parameters, data_len):
    arg_maxs = np.zeros(data_len)
    for i in range(data_len):
        arg_max = 0
        if weight_fun(2 * np.pi, *fitted_parameters[i]) > weight_fun(arg_max, *fitted_parameters[i]):
            arg_max = 2 * np.pi
        phi = np.arctan(fitted_parameters[i][2] / fitted_parameters[i][1])

        if 0 < phi < 2 * np.pi and weight_fun(phi, *fitted_parameters[i]) > weight_fun(arg_max, *fitted_parameters[i]):
            arg_max = phi
        if 0 < phi + np.pi < 2 * np.pi and weight_fun(phi + np.pi, *fitted_parameters[i]) > weight_fun(arg_max, *fitted_parameters[i]):
            arg_max = phi + np.pi
        if 0 < phi + 2 * np.pi < 2 * np.pi and weight_fun(phi + 2 * np.pi, *fitted_parameters[i]) > weight_fun(arg_max,
                                                                                                   *fitted_parameters[i]):
            arg_max = phi + 2 * np.pi
        arg_maxs[i] = arg_max
    return arg_maxs


# here weights and arg_maxs are calculated from continuum distributions
def calc_weights(classes, popts, data_len, num_classes):
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(classes, *popts[i])
    return weights


def preprocess_data(args):
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    reuse_weigths = args.REUSE_WEIGTHS  # Set this flag to true if you want reuse calculated weights

    print("Loading data")
    data = read_np(os.path.join(data_path, "rhorho_raw.data.npy"))
    w = read_np(os.path.join(data_path, "rhorho_raw.w.npy")).swapaxes(0, 1)
    perm = read_np(os.path.join(data_path, "rhorho_raw.perm.npy"))
    print("Read %d events" % data.shape[0])

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

    if not reuse_weigths or not os.path.exists(os.path.join(data_path, 'weigths_{}.npy'.format(num_classes))) \
            or not os.path.exists(os.path.join(data_path, 'arg_maxs.npy')) \
            or np.load(os.path.join(data_path, 'weigths_{}.npy'.format(num_classes))).shape[1] != num_classes:
        weights = calc_weights(classes, popts, data_len, num_classes)
        arg_maxs = calc_arg_maxs(popts, data_len)
        np.save(os.path.join(data_path, 'weigths_{}.npy'.format(num_classes)), weights)
        np.save(os.path.join(data_path, 'arg_maxs.npy'), arg_maxs)
    weights = np.load(os.path.join(data_path, 'weigths_{}.npy'.format(num_classes)))
    arg_maxs = np.load(os.path.join(data_path, 'arg_maxs.npy'))

    #ERW
    # here arg_maxs are in fraction of pi, not in the class index
    # how we go then from fraction of pi to class index??
    # MS: There is not always direct way to go from fraction of pi to class index.
    # MS: arg_max is exact value of most probable CPmix, used in regression type of training.
    # MS: When we want to find class that corresponds to most probable angle
    # MS: we can calculate e.g. np.argmax(weights, axis=1)/(num_class-1)*2*np.pi, which is not equal to exact value
    # print "preprocess: weights", weights
    # print "preprocess: arg_maxs", arg_maxs

    #ERW
    # I am not sure what the purpose is and if it make sense.
    # It makes sense when we choose regression type of training. In classificator it is not used.
    if args.RESTRICT_MOST_PROBABLE_ANGLE:
        arg_maxs[arg_maxs > np.pi] = -1 * arg_maxs[arg_maxs > np.pi] + 2 * np.pi

    if args.NORMALIZE_WEIGHTS:
        weights = weights/np.reshape(popts[:, 0], (-1, 1))

    # ERW
    # here weights and arg_maxs are calculated at value of CPmix representing given class
    # in training, class is expressed as integer, not fraction pf pi.
    if args.WEIGHTS_SUBSET:
        weights_new = np.zeros((weights.shape[0], len(args.WEIGHTS_SUBSET)))
        for i, w in enumerate(args.WEIGHTS_SUBSET):
            weights_new[:, i] = weights[:, w]
        weights = weights_new
    return data, weights, arg_maxs, perm, popts


def calc_min_distances(pred_arg_maxs, calc_arg_maxs, num_class):
    min_distances = np.zeros(len(calc_arg_maxs))
    for i in range(len(calc_arg_maxs)):
        dist = pred_arg_maxs[i] - calc_arg_maxs[i]
        if np.abs(num_class + pred_arg_maxs[i] - calc_arg_maxs[i])<np.abs(dist):
            dist = num_class + pred_arg_maxs[i] - calc_arg_maxs[i]
        if np.abs(-num_class + pred_arg_maxs[i] - calc_arg_maxs[i])<np.abs(dist):
            dist = -num_class + pred_arg_maxs[i] - calc_arg_maxs[i]
        min_distances[i] = dist
    return min_distances


def calculate_metrics(num_class, calc_w, preds_w):
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, num_class))
    preds_w = preds_w / np.tile(np.reshape(np.sum(preds_w, axis=1), (-1, 1)), (1, num_class))
    pred_arg_maxs = np.argmax(preds_w, axis=1)
    calc_arg_maxs = np.argmax(calc_w, axis=1)
    min_distances = calc_min_distances(pred_arg_maxs, calc_arg_maxs, num_class)

    acc0 = (np.abs(min_distances) <= 0).mean()
    acc1 = (np.abs(min_distances) <= 1).mean()
    acc2 = (np.abs(min_distances) <= 2).mean()
    acc3 = (np.abs(min_distances) <= 3).mean()

    mean_error = np.mean(np.abs(min_distances))
    l1_delta_w = np.mean(np.abs(calc_w - preds_w))
    l2_delta_w = np.sqrt(np.mean((calc_w - preds_w) ** 2))

    return np.array([acc0, acc1, acc2, acc3, mean_error, l1_delta_w, l2_delta_w])