import numpy as np

from src_py.metrics_utils import calculate_deltas_signed_pi, calculate_deltas_signed


def test_calculate_deltas_signed_pi():
    first = np.array([0, 1, 2, -1, 1])
    second = np.array([5, 5, 5, 5, 5])
    expected = np.array([1.28318531, 2.28318531, -3., 0.28318531, 2.28318531])
    result = calculate_deltas_signed_pi(first, second)
    assert (np.abs(expected - result) < 0.001).all()


def test_calculate_deltas_signed():
    first = np.array([0, 1, 2, 3, 4, 5, 6])
    second = np.array([6, 5, 4, 3, 2, 1, 0])
    result = calculate_deltas_signed(first, second, 7)
    expected = np.array([-1, -3, 2, 0, -2, 3, 1])
    assert (result == expected).all()


test_calculate_deltas_signed_pi()
test_calculate_deltas_signed()