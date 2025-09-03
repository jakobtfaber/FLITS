from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class PipelineConfig:
    """Container for configuration and CLI options.

    Parameters
    ----------
    data_path: Path to ``.npy`` file containing dynamic spectrum data.
    dm_init: Initial dispersion measure for the fit.
    outdir: Directory where any products will be written.
    steps: Number of sampling steps used by :meth:`fit_models`.
    """

    data_path: Path
    dm_init: float = 0.0
    outdir: Path | None = None
    steps: int = 1000


class BurstFitPipeline:
    """Minimal orchestration for the BurstFit analysis pipeline.

    The class exposes four public stages that can be tested
    independently: :meth:`load_data`, :meth:`fit_models`,
    :meth:`diagnostics`, and :meth:`plot_results`.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset: Dict[str, np.ndarray] | None = None
        self.results: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # pipeline stages
    # ------------------------------------------------------------------
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load the dynamic spectrum from ``config.data_path``.

        Returns
        -------
        dict
            Dictionary containing the 2-D data array along with
            synthetic time and frequency axes.  This structure is
            intentionally light‑weight for ease of testing.
        """

        data = np.load(self.config.data_path)
        time = np.arange(data.shape[1], dtype=float)
        freq = np.arange(data.shape[0], dtype=float)
        self.dataset = {"data": data, "time": time, "freq": freq}
        return self.dataset

    def fit_models(self) -> Dict[str, float]:
        """Perform a trivial "fit" on the loaded data.

        The method currently returns the mean value of the dynamic
        spectrum to act as a place‑holder for a real model fit.
        """

        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call 'load_data' first.")
        mean_val = float(np.mean(self.dataset["data"]))
        self.results = {"mean": mean_val}
        return self.results

    def diagnostics(self) -> Dict[str, float]:
        """Compute simple diagnostic statistics on the data."""

        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call 'load_data' first.")
        std_val = float(np.std(self.dataset["data"]))
        return {"std": std_val}

    def plot_results(self, *, fig: Figure, ax: Axes) -> Tuple[Figure, Axes]:
        """Plot the dynamic spectrum using the provided ``Axes``.

        The function does not create any global matplotlib state; the
        caller is responsible for creating the ``Figure`` and ``Axes``.
        """

        if self.dataset is None:
            raise RuntimeError("Data not loaded. Call 'load_data' first.")
        im = ax.imshow(self.dataset["data"], aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        return fig, ax
