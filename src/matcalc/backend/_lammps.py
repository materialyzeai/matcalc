from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core.structure import Structure

    from ._base import SimulationResult


def run_lammps(
    structure: Structure | Atoms,
    calculator: Calculator,
    *,
    relax_atoms: bool = False,
    relax_cell: bool = False,
    optimizer: str = "FIRE",
    max_steps: int = 500,
    traj_file: str | None = None,
    interval: int = 1,
    fmax: float = 0.1,
    cell_filter: str | None = None,  # type:ignore[assignment]
    cell_filter_kwargs: dict | None = None,
) -> SimulationResult:
    """
    Run LAMMPS calculation using the given structure and calculator.

    Parameters:
    structure (Structure|Atoms): The input structure to calculate potential energy, forces, and stress.
    calculator (Calculator): The calculator object to use for the calculation.

    Returns:
    PESResult: Object containing potential energy, forces, and stress of the input structure.
    """
    raise NotImplementedError("LAMMPS calculation is not yet implemented.")
