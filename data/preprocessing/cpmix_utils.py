import numpy as np
import os
from scipy import optimize


def read_np(filename):
    """ Return the data loaded from a NPY file """
    with open(filename, 'rb') as f:
        return np.load(f)


def weight_function(alphaCP, c0, c1, c2):
    """ Calculate the spin weight by using the alphaCP angle [0, 2*pi] 
    corresponding the scalar state, as well as the C0/C1/C2 coefficients """
    return c0 + c1 * np.cos(alphaCP) + c2 * np.sin(alphaCP)


def one_hot_encode(classes, x):
    """ Assign x to one of the intervals defined by the classes. 
    Return a vector which is the one-hot encoded representation of x assigned
    to a specific bin among all the len(classes) bins available """
    num_classes = len(classes)
    encoded_vector = np.zeros(num_classes)

    for i in range(num_classes - 1):
        if classes[i] <= x < classes[i + 1]:
          encoded_vector[i] = 1.0
    if x >= classes[num_classes - 1]:
        encoded_vector[i] = 1.0
    
    return encoded_vector


def one_hot_encode_c_coefficients(classes, c_coefficients, data_len):
    """ 
    Prepares data for learning the coefficients C0, C1, C2 and drawing 
    their distribution plots.
    
    Parameters:
    - classes: array-like, boundaries defining different classes or bins.
    - c_coefficients: array-like, coefficients for C0, C1, and C2.
    - data_len: int, length of the data.

    Returns:
    - C0 (one_hot_encoded_coefficients[0]): array-like, one-hot encoded data for C0.
    - C1 (one_hot_encoded_coefficients[1]): array-like, one-hot encoded data for C1.
    - C2 (one_hot_encoded_coefficients[2]): array-like, one-hot encoded data for C2.
    """
    num_classes = len(classes)
    one_hot_encoded_coefficients = []
    for i in range(3):
        one_hot_encoded_coefficients.append(np.zeros((data_len, num_classes)))

    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been processed", end='\r')
        for k in range(3):
            if k == 0:
                x = c_coefficients[i][0]
            else:
                x = c_coefficients[i][k] + 1.0
            one_hot_encoded_coefficients[k][i] = one_hot_encode(classes, x)

    return one_hot_encoded_coefficients[0], one_hot_encoded_coefficients[1], \
        one_hot_encoded_coefficients[2]


# here weights and argmaxs are calculated from continuum distributions
def calculate_weights_and_argmaxes(classes, c_coefficients, data_len):
    """
    Calculate weights and argmax values from continuum distributions.

    Parameters:
    - classes: array-like, boundaries defining different classes or bins.
    - c_coefficients: array-like, continuum distributions of C0, C1, and C2 coefficients.
    - data_len: int, length of the data.

    Returns:
    - weights: array-like, weights calculated from continuum distributions for each data point.
    - argmaxes: array-like, argmax values calculated from continuum distributions for each data point.
    - one_hot_encoded_argmaxes: array-like, one-hot encoded representation of argmax values for each data point.
    """
    num_classes = len(classes)
    argmaxes = np.zeros((data_len, 1))
    weights = np.zeros((data_len, num_classes))
    one_hot_encoded_argmaxes = np.zeros((data_len, num_classes))

    print("Calculating weights and argmax values from continuum distributions")    
    for i in range(data_len):
        if i % 10000 == 0:
            print(f"{i} events have been processed...", end='\r')
        weights[i] = weight_function(classes, *c_coefficients[i])
        arg_max = 0
        if weight_function(2 * np.pi, *c_coefficients[i]) > \
            weight_function(arg_max, *c_coefficients[i]):
            arg_max = 2 * np.pi
        phi = np.arctan(c_coefficients[i][2] / c_coefficients[i][1])

        if (0 < phi < 2 * np.pi) and \
            weight_function(phi, *c_coefficients[i]) > weight_function(arg_max, *c_coefficients[i]):
            arg_max = phi

        if (0 < phi + np.pi < 2 * np.pi) and \
            weight_function(phi + np.pi, *c_coefficients[i]) > weight_function(arg_max, *c_coefficients[i]):
            arg_max = phi + np.pi
        
        if (0 < phi + 2 * np.pi < 2 * np.pi) and \
            weight_function(phi + 2 * np.pi, *c_coefficients[i]) > weight_function(arg_max, *c_coefficients[i]):
            arg_max = phi + 2 * np.pi

        argmaxes[i] = arg_max
        one_hot_encoded_argmaxes[i] = one_hot_encode(classes, arg_max)

    return weights, argmaxes, one_hot_encoded_argmaxes


def preprocess_data(args):
    """
    Preprocesses the data for training, including loading, calculating coefficients, 
    and transforming data into suitable formats.

    Parameters:
    - args: Namespace, containing the parsed command-line arguments.

    Returns:
    - data: array-like, containing the loaded data.
    - weights: array-like, containing calculated weights from the continuum distributions.
    - argmaxes: array-like, containing calculated argmax values from the continuum distributions.
    - TODO: perm: ?
    - c_coefficients: array-like, containing calculated C0, C1, and C2 coefficients.
    - ohe_argmaxes: array-like, containing one-hot encoded representations of argmax values.
    - ohe_coefficients: array-like, containing one-hot encoded representations of C0, C1, and C2 coefficients.

    The function preprocesses the data for training by:
    1. Extracting the decay mode suffix from the 'TYPE' argument.
    2. Loading the raw data, weights, and permutations from NPY files.
    3. Calculating C0, C1, and C2 coefficients using curve fitting with scipy.optimize.curve_fit().
    4. Saving the calculated coefficients and covariance matrices as NPY files if not already saved.
    5. Converting the C0, C1, and C2 coefficients into one-hot encoded representations if not already done.
    6. Loading one-hot encoded C0, C1, and C2 coefficients based on user-specified arguments.
    7. Calculating weights and argmaxes from the continuum distributions if not already calculated and saved.
    8. Applying optional data preprocessing steps such as restricting argmax values and normalizing weights.
    9. Returning the preprocessed data including data, weights, argmaxes, TODO: perm ?, 
       C coefficients, one-hot encoded argmaxes, and one-hot encoded coefficients.
    """
    data_path = args.IN
    num_classes = args.NUM_CLASSES
    reuse_weights = args.REUSE_WEIGHTS 
    
    # Extracting the decay mode (e.g. "nn_rhorho" -> "rhorho")
    suffix = (args.TYPE).split("_")[-1] 

    # Reading the data
    print("Loading data")
    data = read_np(os.path.join(data_path, suffix + "_raw.data.npy"))
    weights = read_np(os.path.join(data_path, suffix + "_raw.weights.npy")).swapaxes(0, 1)
    perm = read_np(os.path.join(data_path, suffix + "_raw.perm.npy"))
    print(f"Read {data.shape[0]} events")
    data_len = data.shape[0]

    # Calculating and saving the C coefficients
    c_coefficients_path = os.path.join(data_path, "c_coefficients.npy") 
    c_covariance_path = os.path.join(data_path, 'c_covariance.npy')

    if not os.path.exists(c_coefficients_path):
        # Array to store C0, C1, and C2 coefficients (per event) 
        # It will be the input for the regression or softmax
        c_coefficients = np.zeros((data_len, 3)) 
        
        # Array to store covariance matrices per data point  
        c_covariance  = np.zeros((data_len, 3, 3)) 

        # Values of CPmix at which data were generated
        alphaCP = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]) * np.pi
        
        print("Calculating C0/C1/C2 and the covariance with scipy.optimize.curve_fit()")
        for i in range(data_len):
            if i % 10000 == 0:
                print(f"{i} events have been used by scipy.optimize.curve_fit()", end='\r')
            # Optimize and fit the weight function to the data using curve fitting
            # The 'p0' parameter provides initial guesses for the optimization process
            coeff, ccov = optimize.curve_fit(f=weight_function, xdata=alphaCP, 
                                             ydata=weights[i, :], p0=[1, 1, 1])
            c_coefficients[i]  = coeff
            c_covariance[i]  = ccov

        # Saving the coefficients and covariance as NPY files
        np.save(c_coefficients_path, c_coefficients)
        np.save(c_covariance_path, c_covariance)

    # Loading C0, C1, C2 and saving them in a one-hot encoded form
    c_coefficients = np.load(c_coefficients_path)

    c_paths = []
    for i in range(3):
        c_paths.append(os.path.join(data_path, f'one_hot_encoded_c{i}.npy'))

    if not (os.path.exists(c_paths[0]) \
        and os.path.exists(c_paths[1]) \
        and os.path.exists(c_paths[2]) \
        and np.load(c_paths[0]).shape[1] == num_classes \
        and np.load(c_paths[1]).shape[1] == num_classes \
        and np.load(c_paths[2]).shape[1] == num_classes):
        classes = np.linspace(0, 2, num_classes)
        print("Converting the C0/C1/C1 coefficients to a one-hot encoded format") 
        ohe_coefficients = one_hot_encode_c_coefficients(classes, 
                                                        c_coefficients, 
                                                        data_len)
        print("Saving the C0/C1/C2 coefficients in one-hot encoded form")
        np.save(c_paths[0], ohe_coefficients[0])
        np.save(c_paths[1], ohe_coefficients[1])
        np.save(c_paths[2], ohe_coefficients[2])

    if args.OHE_C_COEFFICIENTS == "one_hot_encoded_c0": 
        ohe_coefficients = np.load(c_paths[0])
    elif args.OHE_C_COEFFICIENTS == "one_hot_encoded_c1":   
        ohe_coefficients = np.load(c_paths[1])
    elif args.OHE_C_COEFFICIENTS == "one_hot_encoded_c2":   
        ohe_coefficients = np.load(c_paths[2])

    # Calculating the weights and argmaxes (one-hot encoded) and saving them
    weights_path = os.path.join(data_path, 'weights.npy')
    argmaxes_path = os.path.join(data_path, 'argmaxes.npy')
    ohe_argmaxes_path = os.path.join(data_path, 'one_hot_encoded_argmaxes.npy')

    if not (reuse_weights and os.path.exists(weights_path) \
        and os.path.exists(argmaxes_path) \
        and os.path.exists(ohe_argmaxes_path) \
        and np.load(weights_path).shape[1] == num_classes \
        and np.load(ohe_argmaxes_path).shape[1] == num_classes):
        classes = np.linspace(0, 2, num_classes) * np.pi
        weights, argmaxes, ohe_argmaxes = calculate_weights_and_argmaxes(classes, 
                                                                         c_coefficients, 
                                                                         data_len)
        np.save(weights_path, weights)
        np.save(argmaxes_path, argmaxes)
        np.save(ohe_argmaxes_path, ohe_argmaxes)
    
    weights  = np.load(weights_path)
    argmaxes = np.load(argmaxes_path)
    ohe_argmaxes = np.load(ohe_argmaxes_path)

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
        argmaxes[argmaxes > np.pi] = -1 * argmaxes[argmaxes > np.pi] + 2 * np.pi

    # Comment from ERW:
    # This optimization process does not provide the expected improvement. 
    # TODO: Revisit the implementation; it's possible that it has not been correctly implemented.
    if args.NORMALIZE_WEIGHTS:
        weights = weights/np.reshape(c_coefficients[:, 0], (-1, 1))
        
    # Comment from ERW:
    # Here, weights and argmax values are calculated at the value of CPmix representing a given class.
    # In training, the class is expressed as an integer, not as a fraction of pi.
    return data, weights, argmaxes, perm, c_coefficients, ohe_argmaxes, ohe_coefficients
