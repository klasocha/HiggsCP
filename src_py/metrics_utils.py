import numpy as np


def calculate_deltas_unsigned(expected, actual, num_classes):
    deltas = np.min(
        np.stack(
            [np.abs(expected - actual), (num_classes - np.abs(expected - actual))]
        ), axis=0)

    return deltas


def calculate_deltas_signed(expected, actual, num_classes):
    deltas = actual - expected
    deltas -= num_classes * (deltas > (num_classes // 2))
    deltas += num_classes * (deltas <= (-num_classes // 2))

    return deltas


def calculate_deltas_signed_pi(expected, actual):
    # Unsigned
    deltas = np.minimum(np.abs(actual - expected), 2 * np.pi - np.abs(actual - expected))
    deltas *= np.sign((np.cos(actual) * np.sin(expected) - np.sin(actual) * np.cos(expected)))
    return deltas