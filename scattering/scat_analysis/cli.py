from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .pipeline import PipelineConfig, BurstFitPipeline


def main(argv: list[str] | None = None) -> None:
    """Entry point for running the BurstFit pipeline from the command line."""

    parser = argparse.ArgumentParser(description="Run BurstFit pipeline")
    parser.add_argument("data_path", type=Path, help="Path to .npy dynamic spectrum")
    parser.add_argument("--dm-init", type=float, default=0.0, dest="dm_init")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = PipelineConfig(
        data_path=args.data_path,
        dm_init=args.dm_init,
        outdir=args.outdir,
        steps=args.steps,
    )
    pipe = BurstFitPipeline(config)
    pipe.load_data()
    pipe.fit_models()
    pipe.diagnostics()
    fig, ax = plt.subplots()
    pipe.plot_results(fig=fig, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
