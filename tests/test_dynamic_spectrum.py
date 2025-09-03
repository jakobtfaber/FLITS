import numpy as np
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'scintillation' / 'scint_analysis'))
from core import DynamicSpectrum


def test_rfi_masking_flags_outliers():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (4, 8))
    data[2, :] += 50  # strong RFI channel
    data[:, 7] += 50  # strong RFI time step
    freqs = np.linspace(1000, 1003, 4)
    times = np.arange(8, dtype=float)
    ds = DynamicSpectrum(data, freqs, times)
    config = {
        'analysis': {
            'rfi_masking': {
                'rfi_downsample_factor': 1,
                'manual_burst_window': [2, 6],
                'freq_threshold_sigma': 2.0,
                'time_threshold_sigma': 2.0,
                'off_burst_buffer': 0,
                'enable_time_domain_flagging': True,
            }
        }
    }
    masked = ds.mask_rfi(config)
    assert masked.power.mask[2].all()
    assert masked.power.mask[:, 7].all()
    assert not masked.power.mask[0, 0]
