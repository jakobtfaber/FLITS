from __future__ import annotations

"""Stage for estimating noise characteristics from off-pulse data."""

from typing import Callable, Optional
import logging
import numpy as np
from numpy.typing import NDArray

from .noise import NoiseDescriptor

log = logging.getLogger(__name__)


class NoiseEstimator:
    """Estimate noise descriptors from raw arrays."""

    def __init__(self, estimator: Callable[[NDArray[np.floating]], NoiseDescriptor]) -> None:
        """Initialise with a noise estimation function."""
        self.estimator = estimator

    def run(self, off_pulse_data: NDArray[np.floating]) -> Optional[NoiseDescriptor]:
        """Return a :class:`NoiseDescriptor` describing *off_pulse_data*.

        Parameters
        ----------
        off_pulse_data : ndarray
            Two-dimensional array of off-pulse samples with shape (time, freq).

        Returns
        -------
        NoiseDescriptor | None
            Estimated descriptor or ``None`` if *off_pulse_data* is empty.
        """
        if off_pulse_data.size == 0:
            log.warning("No off-pulse data provided; skipping noise estimation.")
            return None
        desc = self.estimator(off_pulse_data)
        return desc

