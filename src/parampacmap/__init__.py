from importlib.metadata import PackageNotFoundError, version

from . import models, utils
from .parampacmap import ParamPaCMAP, paramrep_weight_schedule, paramrep_const_schedule, pacmap_weight_schedule

try:
    __version__ = version("parampacmap")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["models", "utils", "ParamPaCMAP"]
