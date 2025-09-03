import numpy as np
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'dispersion'))
from dmphasev2 import DMPhaseEstimator


def test_dm_phase_estimator_recovers_zero_dm():
    n_t, n_ch = 128, 16
    dt = 0.001
    freqs = np.linspace(400, 800, n_ch)
    wf = np.zeros((n_t, n_ch), dtype=complex)
    wf[n_t // 2, :] = 1.0
    dm_grid = np.linspace(-5, 5, 11)
    est = DMPhaseEstimator(wf, freqs, dt, dm_grid, ref="top", n_boot=20)
    dm, dm_err = est.get_dm()
    assert abs(dm) < dm_err
    assert dm_err > 0
