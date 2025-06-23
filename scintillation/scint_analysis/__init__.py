# ==============================================================================
# File: scint_analysis/scint_analysis/__init__.py
# ==============================================================================
# This makes the core classes, main pipeline controller, and key utilities
# easily importable from the top level of the package.

# Core data structures from core.py
from .core import DynamicSpectrum, ACF

# Configuration loading from config.py
from .config import load_config

# Noise modeling and synthesis tools from noise_model.py
from .noise import NoiseDescriptor, estimate_noise_descriptor

# Main pipeline controller from pipeline.py
# Now, when this is imported, the noise_model is already known to the package.
from .pipeline import ScintillationAnalysis

# Key plotting functions from plotting.py
from .plotting import plot_analysis_overview, plot_noise_distribution