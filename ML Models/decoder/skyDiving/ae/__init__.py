"""Top-level package for the AutoEncoder project refactored for SOLID principles."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("skyDiving-ae")
except _metadata.PackageNotFoundError:  # Running from source
    __version__ = "0.0.0"

# Convenience imports
from .config import TrainingConfig  # noqa: F401
from .trainer import Trainer  # noqa: F401 