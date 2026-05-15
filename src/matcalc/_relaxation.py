"""Relaxation properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.filters import Filter
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class RelaxCalc(PropCalc):
    """
    A class to perform structural relaxation calculations using ASE (Atomic Simulation Environment).

    This class facilitates structural relaxation by integrating with ASE tools for
    optimized geometry and/or cell parameters. It enables convergence on forces, stress,
    and total energy, offering customization for relaxation parameters and further
    enabling properties to be extracted from the relaxed structure.

    Attributes:
        calculator: ASE calculator or universal model name.
        optimizer: ASE optimizer name or class.
        max_steps: Maximum optimization steps.
        traj_file: Optional ASE trajectory path.
        interval: Trajectory write interval (steps).
        fmax: Force convergence threshold (eV/Å).
        relax_atoms: Relax atomic positions.
        relax_cell: Relax unit cell (via ``cell_filter``).
        cell_filter: Cell filter class when ``relax_cell`` is True.
        cell_filter_kwargs: Kwargs for ``cell_filter``.
        perturb_distance: Random displacement before relax, or None.
        fix_symmetry: Apply ``FixSymmetry`` when True.
        fix_atoms: Fix all atoms, or list of indices to fix.
        symprec: Symmetry tolerance for ``FixSymmetry``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        optimizer: Optimizer | str = "FIRE",
        max_steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        fmax: float = 0.1,
        relax_atoms: bool = True,
        relax_cell: bool = True,
        cell_filter: Filter = FrechetCellFilter,  # type: ignore[assignment]
        cell_filter_kwargs: dict | None = None,
        perturb_distance: float | None = None,
        fix_symmetry: bool = False,
        fix_atoms: bool | list[int] = False,
        symprec: float = 1e-3,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            optimizer: ASE optimizer name or class.
            max_steps: Maximum relaxation steps.
            traj_file: Optional trajectory output path.
            interval: Steps between trajectory writes.
            fmax: Force convergence criterion (eV/Å).
            relax_atoms: Relax atomic positions.
            relax_cell: Relax cell with ``cell_filter``.
            cell_filter: Cell filter class (default ``FrechetCellFilter``).
            cell_filter_kwargs: Kwargs for the cell filter.
            perturb_distance: Random displacement (Å) before relax, or None.
            fix_symmetry: Constrain symmetry when True.
            fix_atoms: Fix all atoms if True, or indices to fix.
            symprec: Symmetry precision for ``FixSymmetry``.
        """
        self.calculator = calculator  # type: ignore[assignment]

        self.optimizer = optimizer
        self.fmax = fmax
        self.interval = interval
        self.max_steps = max_steps
        self.traj_file = traj_file
        self.relax_cell = relax_cell
        self.relax_atoms = relax_atoms
        self.cell_filter = cell_filter
        self.cell_filter_kwargs = cell_filter_kwargs
        self.perturb_distance = perturb_distance
        self.fix_symmetry = fix_symmetry
        self.fix_atoms = fix_atoms
        self.symprec = symprec

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with ``final_structure``, ``energy`` (eV), ``forces`` (eV/A),
            ``stress`` (eV/A^3), lattice parameters ``a``, ``b``, ``c`` (A), angles
            ``alpha``, ``beta``, ``gamma`` (degrees), ``volume`` (A^3), and
            ``_units`` mapping each numeric output to its unit string.
        """
        result = super().calc(structure)

        structure_in: Structure | Atoms = result["final_structure"]

        if self.perturb_distance is not None:
            # Copy first: ``Structure.perturb`` mutates in place, which would clobber a
            # caller-owned (e.g. session-scoped) Structure passed in by reference.
            structure_in = to_pmg_structure(structure_in).copy().perturb(distance=self.perturb_distance, seed=None)

        atoms = to_ase_atoms(structure_in)
        constraints: list[Any] = []
        if self.fix_atoms:
            indices = list(range(len(atoms))) if isinstance(self.fix_atoms, bool) else [int(i) for i in self.fix_atoms]
            constraints.append(FixAtoms(indices=indices))

        if self.fix_symmetry:
            constraints.append(FixSymmetry(atoms, symprec=self.symprec))

        if constraints:
            atoms.set_constraint(constraints)

        r = run_pes_calc(
            atoms,
            self.calculator,
            relax_atoms=self.relax_atoms,
            relax_cell=self.relax_cell,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
            traj_file=self.traj_file,
            interval=self.interval,
            fmax=self.fmax,
            cell_filter=self.cell_filter,
            cell_filter_kwargs=self.cell_filter_kwargs,
        )

        lattice = r.structure.lattice
        result.update(
            {
                "final_structure": r.structure,
                "energy": r.potential_energy,
                "forces": r.forces,
                "stress": r.stress,
                "a": lattice.a,
                "b": lattice.b,
                "c": lattice.c,
                "alpha": lattice.alpha,
                "beta": lattice.beta,
                "gamma": lattice.gamma,
                "volume": lattice.volume,
                "_units": self._merge_units(
                    result,
                    {
                        "energy": "eV",
                        "forces": "eV/A",
                        "stress": "eV/A^3",
                        "a": "A",
                        "b": "A",
                        "c": "A",
                        "alpha": "degree",
                        "beta": "degree",
                        "gamma": "degree",
                        "volume": "A^3",
                    },
                ),
            }
        )

        return result
