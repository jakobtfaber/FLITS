Quick Start
===========

Crossmatching
-------------

.. code-block:: python

   from crossmatching.toa_utilities import measure_fwhm
   width = measure_fwhm(timeseries, time_resolution=0.001, t_factor=4)

Scattering
----------

.. code-block:: python

   from scattering.scat_analysis.burstfit_pipeline import BurstPipeline
   pipe = BurstPipeline('data.npy', 'outputs', 'FRB', dm_init=500)
   results = pipe.run_full(model_scan=False)

Scintillation
-------------

.. code-block:: python

   from scintillation.scint_analysis.pipeline import ScintillationAnalysis
   analysis = ScintillationAnalysis(config)
   analysis.run()
