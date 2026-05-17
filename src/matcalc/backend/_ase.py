"""
This module implements convenience functions to perform various atomic simulation types, specifically relaxation and
static, using either ASE or LAMMPS. This enables the code to use either for the computation of various properties.
"""

from __future__ import annotations

import contextlib
import io
import logging
from inspect import isclass
from typing import TYPE_CHECKING

import ase
import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.optimize.optimize import Optimizer

from matcalc.utils import to_ase_atoms, to_pmg_structure

from ._base import SimulationResult

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


def is_ase_optimizer(key: str | Optimizer) -> bool:
    """
    Determines whether the given key is an ASE optimizer. A key can
    either be a string representing the name of an optimizer class
    within `ase.optimize` or directly be an optimizer class that
    subclasses `Optimizer`.

    If the key is a string, the function checks whether it corresponds
    to a class in `ase.optimize` that is a subclass of `Optimizer`.

    Args:
        key: String name of an ASE optimizer in ``ase.optimize``, or an ``Optimizer`` subclass.

    Returns:
        True if ``key`` names a valid ASE optimizer class or is an ``Optimizer`` subclass.
    """
    if isclass(key) and issubclass(key, Optimizer):
        return True
    if isinstance(key, str):
        return isclass(obj := getattr(ase.optimize, key, None)) and issubclass(obj, Optimizer)
    return False


VALID_OPTIMIZERS = [key for key in dir(ase.optimize) if is_ase_optimizer(key)]


def get_ase_optimizer(optimizer: str | Optimizer) -> Optimizer:
    """
    Retrieve an ASE optimizer instance based on the provided input. This function accepts either a
    string representing the name of a valid ASE optimizer or an instance/subclass of the Optimizer
    class. If a string is provided, it checks the validity of the optimizer name, and if valid, retrieves
    the corresponding optimizer from ASE. An error is raised if the optimizer name is invalid.

    If an Optimizer subclass or instance is provided as input, it is returned directly.

    Args:
        optimizer: ASE optimizer name (str) or ``Optimizer`` subclass.

    Returns:
        The ASE optimizer class or the given ``Optimizer`` subclass.

    Raises:
        ValueError: If ``optimizer`` is a string not in ``VALID_OPTIMIZERS``.
    """
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer

    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")

    return getattr(ase.optimize, optimizer) if isinstance(optimizer, str) else optimizer


def run_ase(
    structure: Structure | Atoms,
    calculator: Calculator,
    *,
    relax_atoms: bool = False,
    relax_cell: bool = False,
    optimizer: Optimizer | str = "FIRE",
    max_steps: int = 500,
    traj_file: str | None = None,
    interval: int = 1,
    fmax: float = 0.1,
    cell_filter: Filter = FrechetCellFilter,  # type:ignore[assignment]
    cell_filter_kwargs: dict | None = None,
) -> SimulationResult:
    """
    Run ASE static calculation using the given structure and calculator.

    Args:
        structure: Input structure for energy, forces, and stress.
        calculator: ASE calculator attached to the structure.
        relax_atoms: Whether to relax atomic positions.
        relax_cell: Whether to relax the cell (with ``cell_filter``).
        optimizer: Optimizer name or class.
        max_steps: Maximum optimization steps.
        traj_file: Optional trajectory path during relaxation.
        interval: Trajectory write interval.
        fmax: Force convergence criterion (eV/Å).
        cell_filter: Cell filter class when ``relax_cell`` is True.
        cell_filter_kwargs: Extra arguments for the cell filter.

    Returns:
        SimulationResult with structure, energies, forces, and stress.
    """
    atoms = to_ase_atoms(structure)
    atoms.calc = calculator
    cell_filter_kwargs = cell_filter_kwargs or {}
    if relax_atoms:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            original_atoms = atoms
            if relax_cell:
                atoms = cell_filter(atoms, **cell_filter_kwargs)  # type:ignore[operator]
            opt = get_ase_optimizer(optimizer)(atoms)  # type:ignore[operator]
            traj = Trajectory(traj_file, "w", original_atoms) if traj_file is not None else None
            if traj is not None:
                opt.attach(traj, interval=interval)
            opt.run(fmax=fmax, steps=max_steps)
            if traj is not None:
                traj.close()
            if opt.nsteps >= max_steps:
                logger.warning("Maximum steps reached in structure relaxation.")

        if relax_cell:
            atoms = atoms.atoms  # type:ignore[attr-defined]
        if opt.nsteps >= max_steps:
            max_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
            logger.warning(
                "Maximum steps reached in structure relaxation. Max|F| = %s eV/Å",
                max_force,
            )

        return SimulationResult(
            to_pmg_structure(atoms),
            atoms.get_potential_energy(),
            atoms.get_kinetic_energy(),
            atoms.get_total_energy(),
            atoms.get_forces(),
            atoms.get_stress(voigt=False),
        )

    return SimulationResult(
        to_pmg_structure(structure),
        atoms.get_potential_energy(),
        atoms.get_kinetic_energy(),
        atoms.get_total_energy(),
        atoms.get_forces(),
        atoms.get_stress(voigt=False),
    )
