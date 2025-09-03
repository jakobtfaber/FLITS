from __future__ import annotations

"""Utility for managing cache file paths and serialization."""

import pickle
from pathlib import Path
from typing import Any


class CacheManager:
    """Simple pickle-based cache manager.

    Parameters
    ----------
    cache_dir : str
        Directory where cache files are stored.
    burst_id : str
        Identifier used as prefix for cache files.
    """
    def __init__(self, cache_dir: str, burst_id: str) -> None:
        """Initialise the cache manager."""
        self.cache_dir = Path(cache_dir)
        self.burst_id = burst_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        """Return the cache path for a given stage."""
        return self.cache_dir / f"{self.burst_id}_{stage}.pkl"

    def load(self, stage: str) -> Any | None:
        """Load cached object for *stage* if available."""
        p = self.path(stage)
        if p.exists():
            with p.open("rb") as fh:
                return pickle.load(fh)
        return None

    def save(self, stage: str, data: Any) -> None:
        """Store *data* for *stage* in the cache."""
        with self.path(stage).open("wb") as fh:
            pickle.dump(data, fh)

