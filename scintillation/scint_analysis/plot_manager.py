from __future__ import annotations

"""Stage for generating diagnostic plots."""

from typing import Tuple, Optional, Dict
import logging
import os


log = logging.getLogger(__name__)


class PlotManager:
    """Create diagnostic plots for prepared data."""

    def __init__(self, config: dict, *, plotting_module, core_module) -> None:
        """Initialise the plot manager."""
        self.config = config
        self.plotting = plotting_module
        self.core = core_module

    def diagnostic_plots(
        self,
        masked_spectrum,
        burst_lims: Tuple[int, int],
        off_pulse_lims: Tuple[int, int],
        baseline_info: Optional[Dict[str, object]] = None,
    ) -> None:
        """Generate on/off pulse and baseline plots.

        Parameters
        ----------
        masked_spectrum : DynamicSpectrum
            Spectrum to visualise.
        burst_lims : tuple
            On-pulse window indices.
        off_pulse_lims : tuple
            Off-pulse window indices.
        baseline_info : dict | None, optional
            Baseline model information, if available.

        Returns
        -------
        None
        """
        diag_config = self.config.get("pipeline_options", {}).get("diagnostic_plots", {})
        if not diag_config.get("enable", False):
            return

        plot_dir = diag_config.get("directory", "./plots/diagnostics")
        os.makedirs(plot_dir, exist_ok=True)
        burst_id = self.config.get("burst_id", "unknown_burst")

        try:
            on_pulse_power = masked_spectrum.power[:, burst_lims[0]:burst_lims[1]]
            on_pulse_times = masked_spectrum.times[burst_lims[0]:burst_lims[1]]
            on_ds = self.core.DynamicSpectrum(on_pulse_power, masked_spectrum.frequencies, on_pulse_times)
            self.plotting.plot_pulse_window_diagnostic(
                on_ds,
                title="On-Pulse Region",
                save_path=os.path.join(plot_dir, f"{burst_id}_on_pulse_diagnostic.png"),
            )

            off_pulse_power = masked_spectrum.power[:, off_pulse_lims[0]:off_pulse_lims[1]]
            off_pulse_times = masked_spectrum.times[off_pulse_lims[0]:off_pulse_lims[1]]
            off_ds = self.core.DynamicSpectrum(off_pulse_power, masked_spectrum.frequencies, off_pulse_times)
            self.plotting.plot_pulse_window_diagnostic(
                off_ds,
                title="Off-Pulse (Noise) Region",
                save_path=os.path.join(plot_dir, f"{burst_id}_off_pulse_diagnostic.png"),
            )
        except Exception as exc:
            log.error("Failed to generate on/off pulse diagnostic plots: %s", exc)

        if baseline_info:
            try:
                self.plotting.plot_baseline_fit(
                    off_pulse_spectrum=baseline_info["original_data"],
                    fitted_baseline=baseline_info["model"],
                    frequencies=masked_spectrum.frequencies,
                    poly_order=baseline_info["poly_order"],
                    save_path=os.path.join(plot_dir, f"{burst_id}_baseline_diagnostic.png"),
                )
            except Exception as exc:
                log.error("Failed to generate baseline diagnostic plot: %s", exc)

