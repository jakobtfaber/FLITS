from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

__all__ = ["plot_dynamic", "goodness_of_fit"]


def plot_dynamic(
    ax,
    dyn: NDArray[np.floating],
    time: NDArray[np.floating],
    freq: NDArray[np.floating],
    **imshow_kw,
):
    """Imshow wrapper with correct axes."""
    imshow_kw.setdefault("aspect", "auto")
    imshow_kw.setdefault("origin", "lower")
    imshow_kw.setdefault("interpolation", "nearest")
    extent = [time[0], time[-1], freq[0], freq[-1]]
    return ax.imshow(dyn, extent=extent, **imshow_kw)


def goodness_of_fit(
    data: NDArray[np.floating],
    model: NDArray[np.floating],
    noise_std: NDArray[np.floating],
    n_params: int,
) -> Dict[str, Any]:
    """Compute goodness-of-fit metrics."""
    residual = data - model
    noise_std_safe = np.clip(noise_std, 1e-9, None)[:, np.newaxis]

    chi2 = np.sum((residual / noise_std_safe) ** 2)
    ndof = data.size - n_params
    chi2_reduced = chi2 / ndof if ndof > 0 else np.inf

    residual_profile = np.sum(residual, axis=0)
    residual_profile -= np.mean(residual_profile)

    autocorr = np.correlate(residual_profile, residual_profile, mode="same")
    center_val = autocorr[len(autocorr) // 2]
    if center_val > 0:
        autocorr /= center_val

    return {
        "chi2": float(chi2),
        "chi2_reduced": float(chi2_reduced),
        "ndof": int(ndof),
        "residual_rms": float(np.std(residual)),
        "residual_autocorr": autocorr,
    }
