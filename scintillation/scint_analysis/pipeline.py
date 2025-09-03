from __future__ import annotations

"""High level controller for the scintillation analysis pipeline."""

import logging

from . import core, analysis, noise, plotting
from .cache_manager import CacheManager
from .data_preparation import DataPreparation
from .noise_estimator import NoiseEstimator
from .acf_analyzer import ACFAnalyzer
from .plot_manager import PlotManager

log = logging.getLogger(__name__)


class ScintillationAnalysis:
    """Orchestrate the individual pipeline stages."""

    def __init__(self, config: dict) -> None:
        """Initialise the controller with *config*.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the pipeline.
        """
        self.config = config
        self.cache = CacheManager(
            config.get("pipeline_options", {}).get("cache_directory", "./cache"),
            config.get("burst_id", "unknown_burst"),
        )
        self.data_stage = DataPreparation(config, self.cache, core_module=core)
        self.noise_stage = NoiseEstimator(noise.estimate_noise_descriptor)
        self.acf_stage = ACFAnalyzer(config, self.cache, analysis_module=analysis)
        self.plot_manager = PlotManager(config, plotting_module=plotting, core_module=core)

        self.masked_spectrum = None
        self.noise_descriptor = None
        self.acf_results = None
        self.final_results = None
        self.all_subband_fits = None
        self.all_powerlaw_fits = None
        self.intra_pulse_results = None

    def run(self) -> None:
        """Execute the full analysis pipeline.

        Returns
        -------
        None
        """
        (
            self.masked_spectrum,
            burst_lims,
            off_pulse_lims,
            baseline_info,
        ) = self.data_stage.run()

        self.plot_manager.diagnostic_plots(
            self.masked_spectrum, burst_lims, off_pulse_lims, baseline_info
        )

        if self.config.get("analysis", {}).get("noise", {}).get("disable", False):
            log.info("Noise modelling disabled by config.")
            self.noise_descriptor = None
        elif off_pulse_lims[1] > off_pulse_lims[0] + 100:
            off_pulse_data = self.masked_spectrum.power.data[:, off_pulse_lims[0]:off_pulse_lims[1]].T
            self.noise_descriptor = self.noise_stage.run(off_pulse_data)
            if self.noise_descriptor:
                log.info(
                    "Noise characterization complete. Detected kind: '%s'",
                    self.noise_descriptor.kind,
                )
        else:
            log.warning("Not enough pre-burst data for robust noise characterization. Skipping.")
            self.noise_descriptor = None

        (
            self.acf_results,
            self.final_results,
            self.all_subband_fits,
            self.all_powerlaw_fits,
            self.intra_pulse_results,
        ) = self.acf_stage.run(self.masked_spectrum, burst_lims, self.noise_descriptor)

        log.info("--- Pipeline execution finished. ---")

