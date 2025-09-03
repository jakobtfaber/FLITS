from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, Sequence, Tuple

import emcee
import numpy as np
from numpy.typing import NDArray

from .parameters import FRBParams
from .model import FRBModel

__all__ = [
    "FRBFitter",
    "build_priors",
    "compute_bic",
]

DM_POSITIVE = {"c0", "zeta", "tau_1ghz"}
_MIN_POS = 1e-6

log = logging.getLogger(__name__)
if not log.handlers:
    log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def _log_prob_wrapper(theta, model, priors, order, key, log_weight_pos):
    """Module-level function for multiprocessing compatibility."""
    logp = 0.0
    for value, name in zip(theta, order[key]):
        lo, hi = priors[name]
        if not (lo <= value <= hi):
            return -np.inf
    if log_weight_pos:
        for name, v in zip(order[key], theta):
            if name in DM_POSITIVE:
                logp += -np.log(v)
    params = FRBParams.from_sequence(theta, key)
    return model.log_likelihood(params, key) + logp


class FRBFitter:
    """Thin wrapper around ``emcee``."""

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
        log_weight_pos: bool = False,
        **kwargs,
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
        widths = np.array([0.01 * (self.priors[n][1] - self.priors[n][0]) for n in names])
        lower, upper = zip(*(self.priors[n] for n in names))
        widths = np.clip(widths, 1e-6, None)
        walkers = np.random.normal(centre, widths, size=(nwalk, len(names)))
        return np.clip(walkers, lower, upper)

    def sample(self, p0: FRBParams, model_key: str = "M3"):
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


def build_priors(
    init: FRBParams,
    *,
    scale: float = 6.0,
    abs_min: float = _MIN_POS,
    abs_max: Dict[str, float] | None = None,
    log_weight_pos: bool = False,
) -> tuple[Dict[str, Tuple[float, float]], bool]:
    """Build simple linear-space top-hat priors that won't strangle the chain."""
    pri = {}
    ceiling = abs_max or {}
    for name, val in asdict(init).items():
        w = max(scale * max(abs(val), 1e-3), 0.5)
        lower = val - w
        upper = val + w
        if name in DM_POSITIVE:
            lower = max(lower, abs_min)
        if name in ceiling:
            upper = min(upper, ceiling[name])
        pri[name] = (lower, upper)
    return pri, log_weight_pos


def compute_bic(logL_max: float, k: int, n: int) -> float:
    """Bayesian Information Criterion (lower is preferred)."""
    return -2.0 * logL_max + k * np.log(n)
