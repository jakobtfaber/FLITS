"""
burstfit.py
===========

Physics kernel + lightweight MCMC wrapper for modelling **fast-radio-burst
dynamic spectra** with dispersion, intra-channel smearing and thin-screen
scattering.

Public API
----------
* :class:`FRBParams`  – dataclass container for model parameters.
* :class:`FRBModel`   – forward model & Gaussian likelihood.
* :class:`FRBFitter`  – emcee front-end with box priors.
* :func:`compute_bic` – Bayesian Information Criterion helper.
* :func:`build_priors`
* :func:`downsample`
* :func:`plot_dynamic`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Sequence, Tuple

import emcee
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

__all__ = [
    "FRBParams",
    "FRBModel",
    "FRBFitter",
    "compute_bic",
    "build_priors",
    "downsample",
    "plot_dynamic",
    "goodness_of_fit",
]

# ----------------------------------------------------------------------
# Module-level constants
# ----------------------------------------------------------------------
DM_DELAY_MS = 4.148808  # cold-plasma dispersion, ms GHz² (pc cm⁻³)⁻¹
DM_SMEAR_MS = 8.3e-6     # intra-channel smearing, ms GHz⁻³ MHz⁻¹ -> this is a more common formulation
                         # The previous value was likely a typo or for a specific setup.
                         # This standard form is: 8.3 * 1e6 * DM * dnu_MHz / nu_GHz**3 / 1e3
                         # We will apply dnu_MHz later, so this constant is 8.3e-6 ms GHz^3 MHz^-1

# ## FIX ##: Restored the set of physically non-negative parameters.
_POSITIVE = {"c0", "zeta", "tau_1ghz"}        # keep in ONE place only
_MIN_POS  = 1e-6   


log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Log-probability wrapper
# ----------------------------------------------------------------------

# ## REFACTOR ##: This is the correct, module-level wrapper for emcee + multiprocessing.
# It takes the model object itself as an argument, making it stateless and pickleable.
def _log_prob_wrapper(theta, model, priors, order, key, log_weight_pos):
    """Module-level function for multiprocessing compatibility."""
    
    logp = 0.0
    
    # 1. Check priors
    for value, name in zip(theta, order[key]):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf
        
    # optional Jeffreys 1/x weight while *still* sampling x linearly
    if log_weight_pos:
        for name, v in zip(order[key], theta):
            if name in _POSITIVE:
                logp += -np.log(v)          # p(x) ∝ 1/x

    # 2. Compute likelihood
    params = FRBParams.from_sequence(theta, key)
    return model.log_likelihood(params, key)

# ----------------------------------------------------------------------
# Dataclass – model parameters
# ----------------------------------------------------------------------
@dataclass
class FRBParams:
    """Parameter container for the scattering model."""
    c0: float
    t0: float
    gamma: float
    zeta: float = 0.0
    tau_1ghz: float = 0.0

    # ## FIX ##: Added the to_sequence method that was missing.
    def to_sequence(self, model_key: str = "M3") -> Sequence[float]:
        """Pack parameters into a flat sequence for a given model_key."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        return [getattr(self, k) for k in key_map[model_key]]

    # ## FIX ##: Added the @classmethod decorator. This was a critical bug.
    @classmethod
    def from_sequence(
        cls, seq: Sequence[float], model_key: str = "M3"
    ) -> "FRBParams":
        """Unpack a flat param vector according to *model_key*."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        kwargs = {k: v for k, v in zip(key_map[model_key], seq)}
        # fill optional keys with defaults
        kwargs.setdefault("zeta", 0.0)
        kwargs.setdefault("tau_1ghz", 0.0)
        return cls(**kwargs)

# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------
class FRBModel:
    """
    Forward model + (Gaussian) log-likelihood for a dynamic spectrum.
    """
    def __init__(
        self,
        time: NDArray[np.floating],
        freq: NDArray[np.floating],
        *,
        data: NDArray[np.floating] | None = None,
        dm_init: float = 0.0,
        df_MHz: float = 0.03051757812, #Channel width is needed for smearing
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

    # ## REFACTOR ##: Smearing calculation is now more explicit.
    def _smearing_sigma(self, dm: float, zeta: float) -> NDArray[np.floating]:
        """Calculates total Gaussian width from intrinsic pulse width (zeta)
        and intra-channel DM smearing."""
        # DM smearing time (ms) = 8.3e-6 * DM * df_MHz / nu_GHz^3
        sig_dm = DM_SMEAR_MS * dm * self.df_MHz * (self.freq ** -3.0)
        # Add in quadrature with intrinsic width
        return np.hypot(sig_dm, zeta)

    def _estimate_noise(self, off_pulse):
        if self.data is None: return None
        if off_pulse is None:
            q = self.time.size // 4
            idx = np.r_[0:q, -q:0]
        else:
            idx = np.asarray(off_pulse)
        
        # Ensure indices are within bounds
        idx = idx[idx < self.data.shape[1]]

        mad = np.median(
            np.abs(self.data[:, idx] - np.median(self.data[:, idx], axis=1, keepdims=True)),
            axis=1,
        )
        return 1.4826 * np.clip(mad, 1e-6, None) # Use a smaller floor

    def __call__(self, p: FRBParams, model_key: str = "M3") -> NDArray[np.floating]:
        """Return model dynamic spectrum for parameters *p*."""
        ref_freq = np.median(self.freq)
        amp = p.c0 * (self.freq / ref_freq) ** p.gamma
        mu = p.t0 + self._dispersion_delay(0.0)[:, None]

        if model_key in {"M1", "M3"}:
            sig = self._smearing_sigma(self.dm_init, p.zeta)[:, None]
        else:
            sig = self._smearing_sigma(self.dm_init, 0.0)[:, None]

        # Guard against non-physical width
        sig = np.clip(sig, 1e-6, None)

        gauss = (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-0.5 * ((self.time - mu) / sig) ** 2)

        # --- FIX: Implement safe division to prevent NaN ---
        gauss_sum = np.sum(gauss, axis=1, keepdims=True)
        # Clip the sum to a tiny positive number to avoid 0/0 division
        safe_gauss_sum = np.clip(gauss_sum, 1e-30, None)
        gauss_norm = gauss / safe_gauss_sum

        profile = amp[:, None] * gauss_norm

        if model_key in {"M2", "M3"} and p.tau_1ghz > 1e-6:
            alpha = 4.0
            tau = p.tau_1ghz * (self.freq / 1.0) ** (-alpha)
            t_kernel = self.time - self.time[0]

            kernel = np.exp(-t_kernel[None, :] / np.clip(tau, 1e-6, None)[:, None])

            # --- FIX: Implement safe division for the kernel as well ---
            kernel_sum = np.sum(kernel, axis=1, keepdims=True)
            safe_kernel_sum = np.clip(kernel_sum, 1e-30, None)
            kernel_norm = kernel / safe_kernel_sum

            return fftconvolve(profile, kernel_norm, mode="same", axes=1)

        if model_key not in {"M0", "M1", "M2", "M3"}:
            raise ValueError(f"unknown model '{model_key}'")

        return profile

    def log_likelihood(self, p: FRBParams, model: str = "M3") -> float:
        if self.data is None or self.noise_std is None:
            raise RuntimeError("need observed data + noise_std for likelihood")
        
        # Protect against all-zero noise_std if a channel is dead
        noise_std_safe = np.clip(self.noise_std, 1e-9, None)

        resid = (self.data - self(p, model)) / noise_std_safe[:, None]
        return -0.5 * np.sum(resid ** 2)

# ----------------------------------------------------------------------
# Sampler wrapper
# ----------------------------------------------------------------------
class FRBFitter:
    """Thin wrapper around *emcee*."""
    _ORDER = {
        "M0": ("c0", "t0", "gamma"),
        "M1": ("c0", "t0", "gamma", "zeta"),
        "M2": ("c0", "t0", "gamma", "tau_1ghz"),
        "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
    }

    def __init__(
        self,
        model: FRBModel,
        priors: Dict[str, Tuple[float, float]],
        *,
        n_steps: int = 1000,
        n_walkers_mult: int = 8,
        pool=None,
        log_weight_pos=False,
        **kwargs
    ):
        self.model = model
        self.priors = priors
        self.n_steps = n_steps
        self.n_walkers_mult = n_walkers_mult
        self.pool = pool
        self.log_weight_pos = log_weight_pos

    def _init_walkers(self, p0: FRBParams, key: str, nwalk: int):
        names = self._ORDER[key]
        centre = np.array([getattr(p0, n) for n in names])
        # Use 1% of prior range for initial walker ball size
        widths = np.array([0.01 * (self.priors[n][1] - self.priors[n][0]) for n in names])
        lower, upper = zip(*(self.priors[n] for n in names))

        # Ensure widths are not zero
        widths = np.clip(widths, 1e-6, None)

        walkers = np.random.normal(centre, widths, size=(nwalk, len(names)))
        return np.clip(walkers, lower, upper)

    def sample(self, p0: FRBParams, model_key: str = "M3"):
        """Run the MCMC sampler."""
        names = self._ORDER[model_key]
        ndim = len(names)
        nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)

        p_walkers = self._init_walkers(p0, model_key, nwalk)

        sampler = emcee.EnsembleSampler(
            nwalkers=nwalk,
            ndim=ndim,
            log_prob_fn=_log_prob_wrapper,
            args=(self.model, self.priors, self._ORDER, model_key, self.log_weight_pos),
            pool=self.pool,
        )
        sampler.run_mcmc(p_walkers, self.n_steps, progress=True)
        return sampler

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_priors(
    init: "FRBParams",
    *,
    scale: float = 6.0,            # half-width multiplier (wide!)
    abs_min: float = _MIN_POS,     # floor for positive parameters
    abs_max: dict[str, float] | None = None,  # optional hard ceilings
    log_weight_pos: bool = False,  # True → Jeffreys p(x)∝1/x   (still linear sampling)
) -> dict[str, tuple[float, float]]:
    """
    Build simple linear-space top-hat priors that *won’t* strangle the chain.

    Parameters
    ----------
    init
        The optimiser-derived initial parameter set.
    scale
        Half-width multiplier around each init value.
    abs_min
        Lower floor for every positive-definite parameter.
    abs_max
        Optional per-parameter upper caps, e.g. {"tau_1ghz": 1e5}.
    log_weight_pos
        If True, you still sample in linear units but will later *add*
        -log(x) to the log-prior for each positive parameter.
    """
    from dataclasses import asdict
    pri = {}
    ceiling = abs_max or {}
    for name, val in asdict(init).items():
        w     = max(scale * max(abs(val), 1e-3), 0.5)     # ≥ 0.5 half-width
        lower = val - w
        upper = val + w
        if name in _POSITIVE:               # enforce positivity
            lower = max(lower, abs_min)
        if name in ceiling:                 # honour hard caps
            upper = min(upper, ceiling[name])
        pri[name] = (lower, upper)
    return pri, log_weight_pos

def compute_bic(logL_max: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (lower = preferred)."""
    return -2.0 * logL_max + k * np.log(n)

def downsample(data: NDArray[np.floating], f_factor=1, t_factor=1):
    """Block-average by integer factors along (freq, time)."""
    if f_factor == 1 and t_factor == 1:
        return data
    nf, nt = data.shape
    # Ensure dimensions are divisible by factors
    nf_new = nf - (nf % f_factor)
    nt_new = nt - (nt % t_factor)
    d = data[:nf_new, :nt_new].reshape(
        nf_new // f_factor, f_factor, nt_new // t_factor, t_factor
    )
    return d.mean(axis=(1, 3))

def goodness_of_fit(data: NDArray[np.floating],
                   model: NDArray[np.floating],
                   noise_std: NDArray[np.floating],
                   n_params: int) -> Dict[str, Any]:
    """Compute goodness-of-fit metrics."""
    residual = data - model
    noise_std_safe = np.clip(noise_std, 1e-9, None)[:, np.newaxis]
    
    chi2 = np.sum((residual / noise_std_safe) ** 2)
    ndof = data.size - n_params
    chi2_reduced = chi2 / ndof if ndof > 0 else np.inf

    residual_profile = np.sum(residual, axis=0)
    residual_profile -= np.mean(residual_profile)
    
    autocorr = np.correlate(residual_profile, residual_profile, mode='same')
    center_val = autocorr[len(autocorr)//2]
    if center_val > 0:
        autocorr /= center_val

    return {
        'chi2': float(chi2),
        'chi2_reduced': float(chi2_reduced),
        'ndof': int(ndof),
        'residual_rms': float(np.std(residual)),
        'residual_autocorr': autocorr
    }

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
