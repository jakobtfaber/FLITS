Architecture Overview
=====================

The project is organised into specialised modules:

* ``crossmatching`` – tools for aligning time-of-arrival measurements
  from different telescopes.
* ``scattering`` – the BurstFit pipeline for modelling temporal broadening.
* ``scintillation`` – routines to analyse dynamic spectra for scintillation
  parameters.

Each module exposes a pipeline interface that can be executed
independently while sharing common utilities.
