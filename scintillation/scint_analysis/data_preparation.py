from __future__ import annotations

"""Stage for loading data and applying preliminary processing."""

from typing import Tuple, Optional, Dict, TYPE_CHECKING
import logging

from .cache_manager import CacheManager

# core module is injected for easier testing

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .core import DynamicSpectrum


class DataPreparation:
    """Load raw data, mask RFI and subtract baselines."""

    def __init__(self, config: dict, cache: CacheManager, *, core_module) -> None:
        """Create the data preparation stage."""
        self.config = config
        self.cache = cache
        self.core = core_module

    def run(self) -> Tuple["DynamicSpectrum", Tuple[int, int], Tuple[int, int], Optional[Dict[str, object]]]:
        """Prepare the dynamic spectrum and determine burst and noise windows.

        Returns
        -------
        tuple
            ``(masked_spectrum, burst_lims, off_pulse_lims, baseline_info)``
            where ``baseline_info`` is ``None`` if no subtraction was applied.
        """
        masked = self._load_spectrum()
        rfi_config = self.config.get("analysis", {}).get("rfi_masking", {})

        burst_lims = self._determine_burst_window(masked, rfi_config)
        off_pulse_lims = self._determine_off_pulse_window(burst_lims, rfi_config)

        baseline_info = None
        baseline_cfg = self.config.get("analysis", {}).get("baseline_subtraction", {})
        if baseline_cfg.get("enable", False):
            masked, baseline_info = self._subtract_baseline(masked, off_pulse_lims, baseline_cfg.get("poly_order", 1))

        return masked, burst_lims, off_pulse_lims, baseline_info

    # ------------------------------------------------------------------
    def _load_spectrum(self) -> "DynamicSpectrum":
        """Load spectrum from file or cache.

        Returns
        -------
        DynamicSpectrum
            Masked dynamic spectrum.
        """
        cached = self.cache.load("processed_spectrum")
        if cached and not self.config.get("pipeline_options", {}).get("force_recalc", False):
            log.info("Loading cached processed spectrum from %s", self.cache.path("processed_spectrum"))
            return cached

        log.info("Loading and processing raw data...")
        ds_cfg = self.config.get("pipeline_options", {}).get("downsample", {})
        f_factor = int(ds_cfg.get("f_factor", 1))
        t_factor = int(ds_cfg.get("t_factor", 1))
        spectrum = self.core.DynamicSpectrum.from_numpy_file(self.config["input_data_path"]).downsample(f_factor, t_factor)
        masked = spectrum.mask_rfi(self.config)
        if self.config.get("pipeline_options", {}).get("save_intermediate_steps", False):
            self.cache.save("processed_spectrum", masked)
        return masked

    # ------------------------------------------------------------------
    def _determine_burst_window(self, spectrum, rfi_config) -> Tuple[int, int]:
        """Determine on-pulse window in time bins.

        Returns
        -------
        tuple
            Start and end indices of the burst window.
        """
        manual = rfi_config.get("manual_burst_window")
        if manual and len(manual) == 2:
            log.warning("Using manually specified on-pulse window: %s", manual)
            return tuple(manual)
        log.info("Using automated burst detection for on-pulse window.")
        return spectrum.find_burst_envelope(
            thres=rfi_config.get("find_burst_thres", 5.0),
            padding_factor=rfi_config.get("padding_factor", 0.2),
        )

    # ------------------------------------------------------------------
    def _determine_off_pulse_window(self, burst_lims: Tuple[int, int], rfi_config) -> Tuple[int, int]:
        """Determine off-pulse (noise) window.

        Returns
        -------
        tuple
            Start and end indices of the off-pulse region.
        """
        manual = rfi_config.get("manual_noise_window")
        if manual and len(manual) == 2:
            log.warning("Using manually specified off-pulse window: %s", manual)
            return tuple(manual)
        noise_end_bin = burst_lims[0] - 200
        off_pulse = (max(0, noise_end_bin - 500), noise_end_bin)
        log.info("Using automated off-pulse window: %s", off_pulse)
        return off_pulse

    # ------------------------------------------------------------------
    def _subtract_baseline(self, spectrum, off_pulse_lims: Tuple[int, int], poly_order: int) -> Tuple["DynamicSpectrum", Optional[Dict[str, object]]]:
        """Subtract polynomial baseline from spectrum.

        Returns
        -------
        tuple
            ``(spectrum, info)`` where ``info`` describes the fitted baseline or
            is ``None`` if subtraction was skipped.
        """
        if off_pulse_lims[1] <= off_pulse_lims[0] + 50:
            log.warning("Not enough off-pulse data to model baseline. Skipping subtraction.")
            return spectrum, None

        off_pulse_spec = spectrum.get_spectrum(off_pulse_lims)
        spec_before = spectrum
        spectrum, baseline_model = spectrum.subtract_poly_baseline(off_pulse_spec, poly_order=poly_order)
        info = None
        if baseline_model is not None:
            info = {
                "original_data": spec_before.get_spectrum(off_pulse_lims),
                "model": baseline_model,
                "poly_order": poly_order,
            }
        return spectrum, info

