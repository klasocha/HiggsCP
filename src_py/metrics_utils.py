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
