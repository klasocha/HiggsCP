import numpy as np


def calculate_deltas_unsigned(expected, actual, num_classes):
    deltas = np.min(
        np.stack(
            [np.abs(expected - actual), (num_classes - np.abs(expected - actual))]
        ), axis=0)

    return deltas


def calculate_deltas_signed(expected, actual, num_classes):
    deltas = actual - expected
    unsigned_deltas = np.minimum(np.abs(actual - expected), num_classes - np.abs(actual - expected))
    # Unsure
    sign = 1 * ((deltas < num_classes // 2) & (deltas > 0)) - 1 * (deltas >= num_classes // 2) + \
           1 * (deltas <= -num_classes // 2) - ((deltas < 0) & 1 * (deltas > -num_classes // 2))
    return unsigned_deltas * sign


def calculate_deltas_signed_pi(expected, actual):
    # Unsigned
    deltas = np.minimum(np.abs(actual - expected), 2 * np.pi - np.abs(actual - expected))
    deltas *= np.sign(np.sin(expected - actual))
    return deltas