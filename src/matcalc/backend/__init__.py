"""Provides various backends of running simulations."""

from __future__ import annotations

from matcalc.config import SIMULATION_BACKEND

from ._ase import run_ase
from ._base import SimulationResult  # noqa: TC001
from ._lammps import run_lammps


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
    if SIMULATION_BACKEND == "ASE":
        return run_ase(*arg, **kwargs)
    return run_lammps(*arg, **kwargs)
