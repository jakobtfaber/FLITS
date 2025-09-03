import logging
from typing import Optional

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger configured for the FLITS project.

    Parameters
    ----------
    name : str, optional
        Name of the logger to retrieve. Defaults to the root package name.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
    return logging.getLogger(name if name else "flits")
