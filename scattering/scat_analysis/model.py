from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from .parameters import FRBParams

__all__ = [
    "FRBModel",
    "DM_DELAY_MS",
    "DM_SMEAR_MS",
    "downsample",
]

DM_DELAY_MS = 4.148808  # ms GHz^2 (pc cm^-3)^{-1}
DM_SMEAR_MS = 8.3e-6    # ms GHz^-3 MHz^{-1}

log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class FRBModel:
    """Forward model and Gaussian log-likelihood for a dynamic spectrum."""

    def __init__(
        self,
        time: NDArray[np.floating],
        freq: NDArray[np.floating],
        *,
        data: NDArray[np.floating] | None = None,
        dm_init: float = 0.0,
        df_MHz: float = 0.390625,
        beta: float = 2.0,
        noise_std: NDArray[np.floating] | None = None,
        off_pulse: slice | Sequence[int] | None = None,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.freq = np.asarray(freq, dtype=float)
        self.df_MHz = float(df_MHz)

        if data is not None:
            self.data = np.asarray(data, dtype=float)
            if self.data.shape != (self.freq.size, self.time.size):
                raise ValueError("data must have shape (nfreq, ntime)")
        else:
            self.data = None

        self.dm_init = float(dm_init)
        self.beta = float(beta)

        if not np.allclose(np.diff(self.time), self.time[1] - self.time[0]):
            raise ValueError("time axis must be uniform")
        self.dt = self.time[1] - self.time[0]

        if noise_std is None and self.data is not None:
            self.noise_std = self._estimate_noise(off_pulse)
        else:
            self.noise_std = noise_std

    def _dispersion_delay(
        self, dm_err: float = 0.0, ref_freq: float | None = None
    ) -> NDArray[np.floating]:
        if ref_freq is None:
            ref_freq = self.freq.max()
        return DM_DELAY_MS * dm_err * (self.freq ** -self.beta - ref_freq ** -self.beta)

    def _smearing_sigma(self, dm: float, zeta: float) -> NDArray[np.floating]:
        """Gaussian width from intrinsic pulse width and DM smearing."""
        sig_dm = DM_SMEAR_MS * dm * self.df_MHz * (self.freq ** -3.0)
        return np.hypot(sig_dm, zeta)

    def _estimate_noise(self, off_pulse):
        if self.data is None:
            return None
        if off_pulse is None:
            q = self.time.size // 4
            idx = np.r_[0:q, -q:0]
        else:
            idx = np.asarray(off_pulse)
        idx = idx[idx < self.data.shape[1]]
        mad = np.median(
            np.abs(self.data[:, idx] - np.median(self.data[:, idx], axis=1, keepdims=True)),
            axis=1,
        )
        return 1.4826 * np.clip(mad, 1e-6, None)

    def __call__(self, p: FRBParams, model_key: str = "M3") -> NDArray[np.floating]:
        ref_freq = np.median(self.freq)
        amp = p.c0 * (self.freq / ref_freq) ** p.gamma
        mu = p.t0 + self._dispersion_delay(0.0)[:, None]

        if model_key in {"M1", "M3"}:
            sig = self._smearing_sigma(self.dm_init, p.zeta)[:, None]
        else:
            sig = self._smearing_sigma(self.dm_init, 0.0)[:, None]

        sig = np.clip(sig, 1e-6, None)
        gauss = (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-0.5 * ((self.time - mu) / sig) ** 2)
        gauss_sum = np.sum(gauss, axis=1, keepdims=True)
        gauss_norm = gauss / np.clip(gauss_sum, 1e-30, None)
        profile = amp[:, None] * gauss_norm

        if model_key in {"M2", "M3"} and p.tau_1ghz > 1e-6:
            alpha = 4.0
            tau = p.tau_1ghz * (self.freq / 1.0) ** (-alpha)
            t_kernel = self.time - self.time[0]
            kernel = np.exp(-t_kernel[None, :] / np.clip(tau, 1e-6, None)[:, None])
            kernel_sum = np.sum(kernel, axis=1, keepdims=True)
            kernel_norm = kernel / np.clip(kernel_sum, 1e-30, None)
            return fftconvolve(profile, kernel_norm, mode="same", axes=1)

        if model_key not in {"M0", "M1", "M2", "M3"}:
            raise ValueError(f"unknown model '{model_key}'")
        return profile

    def log_likelihood(self, p: FRBParams, model: str = "M3") -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        noise_std_safe = np.clip(self.noise_std, 1e-9, None)
        resid = (self.data - self(p, model)) / noise_std_safe[:, None]
        return -0.5 * np.sum(resid ** 2)


def downsample(data: NDArray[np.floating], f_factor: int = 1, t_factor: int = 1):
    """Block-average by integer factors along (freq, time)."""
    if f_factor == 1 and t_factor == 1:
        return data
    nf, nt = data.shape
    nf_new = nf - (nf % f_factor)
    nt_new = nt - (nt % t_factor)
    d = data[:nf_new, :nt_new].reshape(
        nf_new // f_factor, f_factor, nt_new // t_factor, t_factor
    )
    return d.mean(axis=(1, 3))
