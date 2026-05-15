"""Calculators for materials properties from the potential energy surface."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from ._adsorption import AdsorptionCalc
from ._base import ChainedCalc, PropCalc
from ._elasticity import ElasticityCalc
from ._eos import EOSCalc
from ._gb import GBCalc
from ._interface import InterfaceCalc
from ._lammps import LAMMPSMDCalc
from ._md import MDCalc
from ._neb import MEP, NEBCalc
from ._phonon import PhononCalc
from ._phonon3 import Phonon3Calc
from ._qha import QHACalc
from ._relaxation import RelaxCalc
from ._stability import EnergeticsCalc
from ._surface import SurfaceCalc
from .config import SIMULATION_BACKEND, clear_cache
from .utils import UNIVERSAL_CALCULATOR_NAMES, UNIVERSAL_CALCULATORS, PESCalculator

# Library convention: attach a NullHandler so that matcalc.* loggers are silent
# unless an application explicitly configures logging. Without this, Python
# would emit a "no handlers could be found" warning for any log call.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed

# Provide an alias for loading calculators quickly.
load_up = PESCalculator.load_universal
load_fp = PESCalculator.load_universal

__all__ = [
    "MEP",
    "SIMULATION_BACKEND",
    "UNIVERSAL_CALCULATORS",
    "UNIVERSAL_CALCULATOR_NAMES",
    "AdsorptionCalc",
    "ChainedCalc",
    "EOSCalc",
    "ElasticityCalc",
    "EnergeticsCalc",
    "GBCalc",
    "InterfaceCalc",
    "LAMMPSMDCalc",
    "MDCalc",
    "NEBCalc",
    "PESCalculator",
    "Phonon3Calc",
    "PhononCalc",
    "PropCalc",
    "QHACalc",
    "RelaxCalc",
    "SurfaceCalc",
    "clear_cache",
    "load_fp",
    "load_up",
]
