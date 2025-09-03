# ==============================================================================
# File: scint_analysis/scint_analysis/__init__.py
# ==============================================================================
# This makes the core classes, main pipeline controller, and key utilities
# easily importable from the top level of the package.

# Core data structures from core.py
from .core import DynamicSpectrum, ACF

# Configuration loading from config.py
from .config import load_config

# Noise modeling and synthesis tools
from .noise import NoiseDescriptor, estimate_noise_descriptor

# Stage utilities
from .cache_manager import CacheManager
from .data_preparation import DataPreparation
from .noise_estimator import NoiseEstimator
from .acf_analyzer import ACFAnalyzer
from .plot_manager import PlotManager

# Main pipeline controller
from .pipeline import ScintillationAnalysis

# Key plotting functions
from .plotting import plot_analysis_overview, plot_noise_distribution

