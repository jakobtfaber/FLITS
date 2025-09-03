from __future__ import annotations

"""Stage for computing ACFs and deriving scintillation parameters."""

from typing import Tuple, List, Dict, Optional
import logging

from .cache_manager import CacheManager
from .noise import NoiseDescriptor

log = logging.getLogger(__name__)


class ACFAnalyzer:
    """Compute sub-band ACFs and perform model fitting."""

    def __init__(self, config: dict, cache: CacheManager, *, analysis_module) -> None:
        """Create the analyzer stage."""
        self.config = config
        self.cache = cache
        self.analysis = analysis_module

    def run(
        self,
        spectrum,
        burst_lims: Tuple[int, int],
        noise_desc: Optional[NoiseDescriptor],
    ) -> Tuple[Dict[str, object], Dict[str, object], List[dict], List[dict], Optional[dict]]:
        """Run ACF calculations and fitting.

        Parameters
        ----------
        spectrum : DynamicSpectrum
            The masked dynamic spectrum.
        burst_lims : tuple
            Time-bin limits of the pulse window.
        noise_desc : NoiseDescriptor | None
            Descriptor for off-pulse noise or ``None``.

        Returns
        -------
        tuple
            ``(acf_results, final_results, subband_fits, powerlaw_fits, intra_pulse_results)``
        """
        acf_results = self.cache.load("acf_results")
        force = self.config.get("pipeline_options", {}).get("force_recalc", False)
        if acf_results is None or force:
            log.info("Calculating ACFs for all sub-bands...")
            acf_results = self.analysis.calculate_acfs_for_subbands(
                spectrum, self.config, burst_lims=burst_lims, noise_desc=noise_desc
            )
            if self.config.get("pipeline_options", {}).get("save_intermediate_steps", False):
                self.cache.save("acf_results", acf_results)
                log.info("Saved ACF results to cache: %s", self.cache.path("acf_results"))

        if self.config.get("pipeline_options", {}).get("halt_after_acf", False):
            log.info("'halt_after_acf' is set to True. Halting after ACF computation.")
            return acf_results, {}, [], [], None

        acf_cfg = self.config.get("analysis", {}).get("acf", {})
        intra_results: Optional[dict] = None
        if acf_cfg.get("enable_intra_pulse_analysis", False) and noise_desc is not None:
            intra_results = self.analysis.analyze_intra_pulse_scintillation(
                spectrum, burst_lims, self.config, noise_desc
            )

        final_results, subband_fits, powerlaw_fits = self.analysis.analyze_scintillation_from_acfs(
            acf_results, self.config
        )
        return acf_results, final_results, subband_fits, powerlaw_fits, intra_results

