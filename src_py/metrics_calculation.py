import numpy as np


def calculate_errors_unsigned(expected, actual, num_classes):
    errors = np.min(
        np.stack(
            [np.abs(expected - actual), (num_classes - np.abs(expected - actual))]
        ), axis=0)

    return errors


def calculate_error_histogram(expected, actual, num_classes):
    errors = calculate_errors_signed(expected, actual, num_classes)

    return np.histogram(errors, bins=range(1 - num_classes // 2, num_classes // 2 + 1, 1))


def calculate_abs_error_histogram(expected, actual, num_classes):
    errors = calculate_errors_unsigned(expected, actual, num_classes)

    return np.histogram(errors, bins=range(0, num_classes))


def calculate_errors_signed(expected, actual, num_classes):
    errors = actual - expected
    errors -= num_classes * (errors > (num_classes // 2))
    errors += num_classes * (errors <= (-num_classes // 2))

    return errors
