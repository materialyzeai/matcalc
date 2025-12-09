"""Calculators for materials properties."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed

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
from .utils import UNIVERSAL_CALCULATORS, PESCalculator

# Provide an alias for loading calculators quickly.
load_up = PESCalculator.load_universal
load_fp = PESCalculator.load_universal
