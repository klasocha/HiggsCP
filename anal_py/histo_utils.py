import numpy as np

from src_py.metrics_utils import calculate_deltas_signed, calculate_deltas_unsigned


def calculate_error_histogram(expected, actual, num_classes):
    errors = calculate_deltas_signed(expected, actual, num_classes)

    return np.histogram(errors, bins=range(1 - num_classes // 2, num_classes // 2 + 1, 1))


def calculate_abs_error_histogram(expected, actual, num_classes):
    errors = calculate_deltas_unsigned(expected, actual, num_classes)

    return np.histogram(errors, bins=range(0, num_classes))