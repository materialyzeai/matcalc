"""Provides various backends of running simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matcalc.config import SIMULATION_BACKEND
from matcalc.utils import PESCalculator

from ._ase import run_ase
from ._base import SimulationResult  # noqa: TC001
from ._lammps import (
    LAMMPSCalculator,
    LAMMPSPotential,
    get_lammps_potential,
    load_lammps_calculator,
    run_lammps,
    to_lammps_calculator,
)
from ._lammps_md import run_lammps_md

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


def is_lammps_backend() -> bool:
    """Whether calculations run with the LAMMPS backend (``MATCALC_BACKEND`` is not ``ASE``)."""
    return SIMULATION_BACKEND != "ASE"


def to_backend_calculator(calculator: str | Calculator) -> Calculator:
    """Return the calculator to use with the active backend.

    Under the ASE backend, a model name is loaded with ``PESCalculator.load_universal`` and a calculator is returned
    unchanged. Under the LAMMPS backend the result is a :class:`LAMMPSCalculator`, so that the force field is resolved
    once (e.g. a MatGL model exported to TorchScript), unsupported models fail immediately, and code that evaluates the
    calculator directly through ASE also uses LAMMPS: a model name is built directly with
    :func:`~matcalc.backend._lammps.load_lammps_calculator` (no ASE calculator is created), and a calculator object is
    converted with :func:`~matcalc.backend._lammps.to_lammps_calculator`.

    Args:
        calculator: Model name or ASE calculator.

    Returns:
        Calculator for the active backend.

    Raises:
        TypeError: Under the LAMMPS backend, if the model or calculator has no LAMMPS representation.
    """
    if not is_lammps_backend():
        return PESCalculator.load_universal(calculator) if isinstance(calculator, str) else calculator
    if isinstance(calculator, str):
        return load_lammps_calculator(calculator)
    return to_lammps_calculator(calculator)


def run_pes_calc(*arg, **kwargs) -> SimulationResult:  # noqa:ANN002,ANN003
    """
    Executes the potential energy surface (PES) calculation using the appropriate backend.

    This function determines the backend to use for the PES calculation based on the
    environment variable `MATCALC_BACKEND`. If the variable is set to `"ASE"` (case-insensitive)
    or is not explicitly provided, the ASE backend is used. Otherwise, the LAMMPS backend is
    selected. The function then forwards the arguments and keyword arguments to the respective
    backend function for execution.

    Args:
        *arg: Positional arguments forwarded to the selected backend.
        **kwargs: Keyword arguments forwarded to the selected backend.

    Returns:
        SimulationResult from the ASE or LAMMPS backend.
    """
    if not is_lammps_backend():
        return run_ase(*arg, **kwargs)
    return run_lammps(*arg, **kwargs)
