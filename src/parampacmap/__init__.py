from importlib.metadata import PackageNotFoundError, version

from . import models, utils
from .parampacmap import ParamPaCMAP

try:
    __version__ = version("parampacmap")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["models", "utils", "ParamPaCMAP"]
