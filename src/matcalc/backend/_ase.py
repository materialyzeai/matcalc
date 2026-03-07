"""
This module implements convenience functions to perform various atomic simulation types, specifically relaxation and
static, using either ASE or LAMMPS. This enables the code to use either for the computation of various properties.
"""

from __future__ import annotations

import contextlib
import io
import pickle
import warnings
from dataclasses import dataclass, field
from inspect import isclass
from typing import TYPE_CHECKING

import ase
from ase.filters import FrechetCellFilter
from ase.optimize.optimize import Optimizer

from matcalc.utils import to_ase_atoms, to_pmg_structure

from ._base import SimulationResult

if TYPE_CHECKING:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from numpy.typing import NDArray
    from pymatgen.core.structure import Structure


def is_ase_optimizer(key: str | Optimizer) -> bool:
    """
    Determines whether the given key is an ASE optimizer. A key can
    either be a string representing the name of an optimizer class
    within `ase.optimize` or directly be an optimizer class that
    subclasses `Optimizer`.

    If the key is a string, the function checks whether it corresponds
    to a class in `ase.optimize` that is a subclass of `Optimizer`.

    :param key: The key to check, either a string name of an ASE
        optimizer class or a class object that potentially subclasses
        `Optimizer`.
    :return: True if the key is either a string corresponding to an
        ASE optimizer subclass name in `ase.optimize` or a class that
        is a subclass of `Optimizer`. Otherwise, returns False.
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

    :param optimizer: The optimizer to be retrieved. Can be a string representing a valid ASE
        optimizer name or an instance/subclass of the Optimizer class.
    :type optimizer: str | Optimizer

    :return: The corresponding ASE optimizer instance or the input Optimizer instance/subclass.
    :rtype: Optimizer

    :raises ValueError: If the optimizer name provided as a string is not among the valid ASE
        optimizer names defined by `VALID_OPTIMIZERS`.
    """
    if isclass(optimizer) and issubclass(optimizer, Optimizer):
        return optimizer

    if optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}")

    return getattr(ase.optimize, optimizer) if isinstance(optimizer, str) else optimizer


@dataclass
class TrajectoryObserver:
    """
    Handles the observation and tracking of the trajectory-related properties of an Atoms object.

    The class captures and stores various physical properties of an Atoms object during relaxation steps
    or a simulation. It allows for the extraction of property snapshots, slicing trajectory data,
    and saving the stored data to a file for further analysis or reuse.

    Attributes:
        atoms: Atoms
            The instance of Atoms to observe and track properties for.
        potential_energies: list[float]
            List to store the potential energies recorded during the trajectory.
        kinetic_energies: list[float]
            List to store the kinetic energies recorded during the trajectory.
        total_energies: list[float]
            List to store the total energies recorded during the trajectory.
        forces: list[np.ndarray]
            List to store the atomic forces measured during the trajectory.
        stresses: list[np.ndarray]
            List to store the stress tensors recorded during the trajectory.
        atom_positions: list[np.ndarray]
            List to store the atomic positions recorded during the trajectory.
        cells: list[np.ndarray]
            List to store the simulation cell geometries recorded during the trajectory.
    """

    atoms: Atoms
    potential_energies: list[float] = field(default_factory=list)
    kinetic_energies: list[float] = field(default_factory=list)
    total_energies: list[float] = field(default_factory=list)
    forces: list[NDArray[np.float64]] = field(default_factory=list)
    stresses: list[NDArray[np.float64]] = field(default_factory=list)
    atom_positions: list[NDArray[np.float64]] = field(default_factory=list)
    cells: list[NDArray[np.float64]] = field(default_factory=list)

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.potential_energies.append(float(self.atoms.get_potential_energy()))
        self.kinetic_energies.append(float(self.atoms.get_kinetic_energy()))
        self.total_energies.append(float(self.atoms.get_total_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        return len(self.total_energies)

    def get_slice(self, sl: slice) -> TrajectoryObserver:
        """
        Gets a sliced view of the trajectory data encapsulated within a new TrajectoryObserver
        object, based on the provided slice. This method extracts the corresponding segment of
        the trajectory data for all relevant attributes and returns a new TrajectoryObserver
        object containing the sliced values.

        Args:
            sl (slice): The slice object specifying the indices to extract from the trajectory
                data.

        Returns:
            TrajectoryObserver: A new TrajectoryObserver instance containing the sliced
                trajectory data.
        """
        return TrajectoryObserver(
            self.atoms,
            self.potential_energies[sl],
            self.kinetic_energies[sl],
            self.total_energies[sl],
            self.forces[sl],
            self.stresses[sl],
            self.atom_positions[sl],
            self.cells[sl],
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "potential_energies": self.potential_energies,
            "kinetic_energies": self.kinetic_energies,
            "total_energies": self.total_energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


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
) -> SimulationResult:
    """
    Run ASE static calculation using the given structure and calculator.

    Parameters:
    structure (Structure|Atoms): The input structure to calculate potential energy, forces, and stress.
    calculator (Calculator): The calculator object to use for the calculation.

    Returns:
    PESResult: Object containing potential energy, forces, and stress of the input structure.
    """
    atoms = to_ase_atoms(structure)
    atoms.calc = calculator
    if relax_atoms:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = cell_filter(atoms)  # type:ignore[operator]
            opt = get_ase_optimizer(optimizer)(atoms)  # type:ignore[operator]
            opt.attach(obs, interval=interval)
            dyn = opt.run(fmax=fmax, steps=max_steps)
            if dyn.nsteps >= max_steps:
                warnings.warn("Maximum steps reached in structure relaxation.", UserWarning, stacklevel=2)
            if traj_file is not None:
                obs()
                obs.save(traj_file)
        if relax_cell:
            atoms = atoms.atoms  # type:ignore[attr-defined]
        return SimulationResult(
            to_pmg_structure(atoms),
            obs.potential_energies[-1],
            obs.kinetic_energies[-1],
            obs.total_energies[-1],
            obs.forces[-1],
            obs.stresses[-1],
        )

    return SimulationResult(
        to_pmg_structure(structure),
        atoms.get_potential_energy(),
        atoms.get_kinetic_energy(),
        atoms.get_total_energy(),
        atoms.get_forces(),
        atoms.get_stress(voigt=False),
    )
