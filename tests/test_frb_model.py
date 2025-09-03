import numpy as np
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'scattering' / 'scat_analysis'))
from burstfit import FRBModel, FRBParams


def test_frb_model_forward_gaussian_normalized():
    time = np.linspace(-1.0, 1.0, 201)
    freq = np.linspace(1.0, 1.5, 4)
    model = FRBModel(time, freq, dm_init=0.0)
    params = FRBParams(c0=1.0, t0=0.0, gamma=0.0, zeta=0.1, tau_1ghz=0.0)
    spec = model(params, model_key="M1")
    assert spec.shape == (freq.size, time.size)
    sums = spec.sum(axis=1)
    assert np.allclose(sums, np.full(freq.size, params.c0), rtol=1e-5)
    peak_indices = spec.argmax(axis=1)
    t_idx = np.argmin(np.abs(time - params.t0))
    assert np.all(np.abs(peak_indices - t_idx) <= 1)
