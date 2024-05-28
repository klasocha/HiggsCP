"""  We quantify the DNN performance for classification problem in the context of physics 
relevant criteria. The first question is how well DNN is able to reproduce per-event shape of 
the spin weight wt_norm. We use L2 norm for that. The second criterium is the difference 
between most probable predicted class and most probable true class, denoted as ∆class.
This module implements the second criterium. """
import numpy as np


def calculate_deltas_unsigned(expected, actual, num_classes):
    """ Calculate the unsigned difference returned by the calculate_deltas_signed() """
    deltas = np.min(np.stack([np.abs(expected - actual), 
                              (num_classes - np.abs(expected - actual))]), axis=0)
    return deltas


def calculate_deltas_signed(expected, actual, num_classes):
    """ Calculate the difference (with the sign) between the most probable predicted class and 
    the most probable true class, denoted as ∆class. When calculating the difference 
    between class indices, periodicity of the spin weight functional form is
    taken into account. Class indices represent discrete values of αCP, in range (0,2π). 
    Thus, the first and last class are the same class -> OK for using in range (0, 2 pi) 
    and trygonometric function not OK otherwise. """
    deltas = actual - expected
    deltas -= (num_classes-1) * (deltas > (num_classes // 2))
    deltas += (num_classes-1) * (deltas <= (-num_classes // 2))
    return deltas


# by ERW
def calculate_deltas_signed_pi(expected, actual):
    """ Calculate the difference similar to the one returned by 
    calculate_deltas_signed() but expressed in radians. """
    deltas = actual - expected
    deltas -= np.pi * (deltas > np.pi)
    deltas += np.pi * (deltas <= -np.pi)
    return deltas


# by J.Kurek
def calculate_deltas_signed_pi_topo(expected, actual):
    """ Calculate the difference similar to the one returned by 
    calculate_deltas_signed() but expressed in radians. """

    # Shifting the predictions to the range [0, 2pi]
    for i in range(len(actual)):
        while actual[i] > (2 * np.pi):
            actual[i] -= 2 * np.pi
        while actual[i] < 0:
            actual[i] += 2 * np.pi

    deltas = np.minimum(np.abs(actual - expected), 2 * np.pi - np.abs(actual - expected))
    deltas *= np.sign(np.sin(expected - actual))
    return deltas