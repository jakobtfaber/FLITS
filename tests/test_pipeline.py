from __future__ import annotations

import sys
from pathlib import Path
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Ensure the repository root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scattering.scat_analysis import PipelineConfig, BurstFitPipeline


def _make_data(tmp_path: Path) -> tuple[Path, np.ndarray]:
    arr = np.random.random((4, 4))
    path = tmp_path / "data.npy"
    np.save(path, arr)
    return path, arr


def test_load_data(tmp_path):
    path, arr = _make_data(tmp_path)
    cfg = PipelineConfig(data_path=path)
    pipe = BurstFitPipeline(cfg)
    ds = pipe.load_data()
    assert np.allclose(ds["data"], arr)


def test_fit_models(tmp_path):
    path, arr = _make_data(tmp_path)
    pipe = BurstFitPipeline(PipelineConfig(data_path=path))
    pipe.load_data()
    res = pipe.fit_models()
    assert np.isclose(res["mean"], np.mean(arr))


def test_diagnostics(tmp_path):
    path, arr = _make_data(tmp_path)
    pipe = BurstFitPipeline(PipelineConfig(data_path=path))
    pipe.load_data()
    diag = pipe.diagnostics()
    assert np.isclose(diag["std"], np.std(arr))


def test_plot_results(tmp_path):
    path, _ = _make_data(tmp_path)
    pipe = BurstFitPipeline(PipelineConfig(data_path=path))
    pipe.load_data()
    fig, ax = plt.subplots()
    fig, returned_ax = pipe.plot_results(fig=fig, ax=ax)
    # The plotting function should use the provided axes
    assert returned_ax is ax
