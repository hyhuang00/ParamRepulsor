from . import models
from . import utils
from .parampacmap import ParamPaCMAP

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('parampacmap')
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["models", "utils", "ParamPaCMAP"]

