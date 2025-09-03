from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["FRBParams"]


@dataclass
class FRBParams:
    """Parameter container for the scattering model."""

    c0: float
    t0: float
    gamma: float
    zeta: float = 0.0
    tau_1ghz: float = 0.0

    def to_sequence(self, model_key: str = "M3") -> Sequence[float]:
        """Pack parameters into a flat sequence for a given ``model_key``."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        return [getattr(self, k) for k in key_map[model_key]]

    @classmethod
    def from_sequence(cls, seq: Sequence[float], model_key: str = "M3") -> "FRBParams":
        """Unpack a flat parameter vector according to ``model_key``."""
        key_map = {
            "M0": ("c0", "t0", "gamma"),
            "M1": ("c0", "t0", "gamma", "zeta"),
            "M2": ("c0", "t0", "gamma", "tau_1ghz"),
            "M3": ("c0", "t0", "gamma", "zeta", "tau_1ghz"),
        }
        kwargs = {k: v for k, v in zip(key_map[model_key], seq)}
        kwargs.setdefault("zeta", 0.0)
        kwargs.setdefault("tau_1ghz", 0.0)
        return cls(**kwargs)
