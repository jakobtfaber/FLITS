"""Generic signal-processing utilities used across FLITS.

This module provides small, dependency-light helpers that operate on
NumPy arrays.  They are intentionally written to work with both plain and
masked arrays.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["downsample", "subtract_baseline"]

def downsample(data: ArrayLike, t_factor: int = 1, f_factor: int = 1) -> NDArray:
    """Block-average an array along time (and optionally frequency) axes.

    Parameters
    ----------
    data : array_like
        Input 1D or 2D array.  For 2D arrays the expected orientation is
        ``(nfreq, ntime)``.
    t_factor : int, optional
        Integer factor by which to downsample the time axis.  Must be ≥1.
    f_factor : int, optional
        Integer factor by which to downsample the frequency axis.  Ignored
        for 1D input.  Must be ≥1.

    Returns
    -------
    ndarray
        Downsampled array.

    Raises
    ------
    ValueError
        If the input is not 1D or 2D, or if any factor is < 1.
    """
    if t_factor < 1 or f_factor < 1:
        raise ValueError("Downsampling factors must be ≥1")

    arr = np.asanyarray(data)

    if arr.ndim == 1:
        nt = arr.shape[0]
        nt_trim = nt - (nt % t_factor)
        if nt_trim == 0:
            result = arr.copy()
        else:
            result = arr[:nt_trim].reshape(nt_trim // t_factor, t_factor).mean(axis=1)
        return np.ma.array(result, copy=False) if np.ma.isMaskedArray(result) else np.asarray(result)

    if arr.ndim == 2:
        nfreq, nt = arr.shape
        nt_trim = nt - (nt % t_factor)
        arr_t = arr[:, :nt_trim].reshape(nfreq, nt_trim // t_factor, t_factor).mean(axis=2)

        nfreq = arr_t.shape[0]
        nf_trim = nfreq - (nfreq % f_factor)
        if nf_trim == 0:
            result = arr_t.copy()
        else:
            result = arr_t[:nf_trim].reshape(nf_trim // f_factor, f_factor, arr_t.shape[1]).mean(axis=1)
        return np.ma.array(result, copy=False) if np.ma.isMaskedArray(result) else np.asarray(result)

    raise ValueError(f"Unsupported array shape {arr.shape}; expected 1D or 2D")

def subtract_baseline(data: ArrayLike, baseline: ArrayLike | None = None, *, axis: int | None = None) -> NDArray:
    """Subtract a baseline from ``data``.

    Parameters
    ----------
    data : array_like
        Input array.
    baseline : array_like, optional
        Baseline values to subtract.  If ``None`` (default) the baseline is
        estimated as the mean along ``axis``.
    axis : int, optional
        Axis along which to compute the mean when ``baseline`` is ``None``.

    Returns
    -------
    ndarray
        Array with the baseline removed.  The returned array is always a
        NumPy array (not a view) to avoid modifying the input in-place.
    """
    arr = np.asanyarray(data)

    if baseline is None:
        baseline = arr.mean(axis=axis, keepdims=True)
    else:
        baseline = np.asanyarray(baseline)
    result = arr - baseline
    return np.ma.array(result, copy=False) if np.ma.isMaskedArray(result) else np.asarray(result)
