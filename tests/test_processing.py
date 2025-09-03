import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flits.signal.processing import downsample, subtract_baseline


def test_downsample_1d():
    data = np.arange(10, dtype=float)
    out = downsample(data, t_factor=2)
    expected = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
    assert np.allclose(out, expected)


def test_downsample_2d():
    data = np.arange(24, dtype=float).reshape(4, 6)
    out = downsample(data, f_factor=2, t_factor=3)
    temp = data.reshape(4, 2, 3).mean(axis=2)
    expected = temp.reshape(2, 2, 2).mean(axis=1)
    assert np.allclose(out, expected)


def test_subtract_baseline_1d():
    data = np.arange(5, dtype=float) + 10
    out = subtract_baseline(data)
    assert np.allclose(out, data - data.mean())
    assert np.isclose(out.mean(), 0.0)


def test_subtract_baseline_2d_axis():
    data = np.arange(12, dtype=float).reshape(3, 4)
    out = subtract_baseline(data, axis=1)
    expected = data - data.mean(axis=1, keepdims=True)
    assert np.allclose(out, expected)
