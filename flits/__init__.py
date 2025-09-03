"""FLITS: FRB Intensity Analysis Toolkit."""

from .scattering.run_scat_analysis import main as run_pipeline
from .scattering.scat_analysis.burstfit_pipeline import BurstPipeline

__all__ = ["run_pipeline", "BurstPipeline"]
