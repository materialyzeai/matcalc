"""Grain Boundary Energy calculations."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core.interface import GrainBoundaryGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class GBCalc(PropCalc):
    """
    A class for performing grain boundary energy calculations by generating grain boundary
    structures from a bulk structure using pymatgen's GrainBoundaryGenerator, optionally
    relaxing bulk and GB, and computing gamma_GB via (E_GB - N_GB * E_bulk_per_atom) / (2 * A).

    Attributes:
        calculator: ASE Calculator used for energy and force evaluations.
        relax_bulk: Whether to relax the bulk structure (cell and atoms) before GB generation.
        relax_gb: Whether to relax the grain boundary structures (atoms only) after generation.
        fmax: Force convergence criterion (eV/Å) for relaxations.
        optimizer: Optimizer for structure relaxations.
        max_steps: Maximum optimization steps for relaxations.
        relax_calc_kwargs: Additional keyword arguments for relaxation calculations.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_bulk: bool = True,
        relax_gb: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "FIRE",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE Calculator or string identifier for a universal calculator.
            relax_bulk: Whether to relax the bulk structure before GB generation. Default True.
            relax_gb: Whether to relax the GB structures after generation. Default True.
            fmax: Force tolerance (eV/Å) for relaxations. Default 0.1.
            optimizer: ASE optimizer or name for relaxations. Default "FIRE".
            max_steps: Max optimization steps. Default 500.
            relax_calc_kwargs: Additional keyword arguments for relaxation calculations. Defaults to None.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.relax_bulk = relax_bulk
        self.relax_gb = relax_gb
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

    def calc_gb(
        self,
        structure: Structure | Atoms,
        sigma: int,
        rotation_axis: tuple[int, int, int],
        gb_plane: tuple[int, int, int],
        rotation_angle: float,
        expand_times: int = 4,
        vacuum_thickness: float = 0.0,
        ab_shift: tuple[float, float] = (0.0, 0.0),
        rotation_angle_tolerance: float = 0.1,
        normal: bool = True,  # noqa: FBT001,FBT002
        rm_ratio: float = 0.7,
        gb_generator_kwargs: dict | None = None,
        gb_from_parameters_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """
        Generate grain boundary structures from bulk, relax them, and compute GB energies.

        Args:
            structure: Bulk Structure for GB generation.
            sigma: Sigma value of the coincident site lattice (CSL) grain boundary.
            rotation_axis: Rotation axis vector for GB (u, v, w).
            gb_plane: Miller indices of GB plane.
            rotation_angle: Rotation angle in degrees.
            expand_times: Multiplier for cell expansion to avoid interactions. Default 4.
            vacuum_thickness: Vacuum layer thickness (Å) between grains. Default 0.0.
            ab_shift: In-plane shift along a, b vectors. Default (0.0, 0.0).
            rotation_angle_tolerance: Tolerance (degrees) for matching rotation angle. Default 0.1.
            normal: Whether the c axis of top grain should be perpendicular to the surface.
                Default True.
            rm_ratio: Criteria to remove atoms which are too close with each other. Default 0.7.
            gb_generator_kwargs: Additional args for GrainBoundaryGenerator(). Default None.
            gb_from_parameters_kwargs: Additional args for gb_from_parameters(). Default None.

        Returns:
            A dictionary containing the updated structure data, including fields like
            'grain_boundary', 'final_grain_boundary', 'gb_relax_energy',
            'grain_boundary_energy', and possibly updated 'bulk_energy_per_atom' and 'final_bulk'.

        Raises:
            ValueError: If no rotation angle within ``rotation_angle_tolerance`` of
                ``rotation_angle`` exists for the given ``sigma``.
        """
        # GB generation - start with a conventional cell for relaxation.
        structure = to_pmg_structure(structure)
        bulk_conventional = structure.to_conventional()

        # Bulk relaxation
        relax_bulk_calc = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=self.relax_bulk,
            relax_atoms=self.relax_bulk,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )

        bulk_opt = relax_bulk_calc.calc(bulk_conventional)
        final_bulk = bulk_opt["final_structure"]
        bulk_energy_per_atom = bulk_opt["energy"] / len(final_bulk)

        # Use relaxed primitive so GB lattice parameters are consistent with the bulk energy.
        bulk_primitive = final_bulk.to_primitive()
        gb_gen = GrainBoundaryGenerator(
            initial_structure=bulk_primitive,
            **(gb_generator_kwargs or {}),
        )

        analyzer = SpacegroupAnalyzer(bulk_primitive)
        lattice_type = analyzer.get_crystal_system()[0]
        c_a_ratio = gb_gen.get_ratio()
        angles = gb_gen.get_rotation_angle_from_sigma(
            sigma=sigma,
            r_axis=rotation_axis,
            lat_type=lattice_type,
            ratio=tuple(c_a_ratio) if c_a_ratio is not None else None,  # type: ignore[arg-type]
        )

        # look through the list of computed angles and find the one
        # that is within the tolerance of the requested angle
        rotation_angle_exact = next(
            (a for a in angles if math.isclose(a, rotation_angle, rel_tol=0, abs_tol=rotation_angle_tolerance)),
            None,
        )
        # …and if none are within the tolerance, error out
        if rotation_angle_exact is None:
            raise ValueError(
                f"No matching rotation angle {rotation_angle} for sigma={sigma}. Possible angles: {angles}"
            )

        grain_boundary = gb_gen.gb_from_parameters(
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle_exact,
            plane=gb_plane,
            expand_times=expand_times,
            vacuum_thickness=vacuum_thickness,
            ab_shift=ab_shift,
            normal=normal,
            rm_ratio=rm_ratio,
            **(gb_from_parameters_kwargs or {}),
        )

        return self.calc(
            {
                "grain_boundary": grain_boundary,
                "bulk_energy_per_atom": bulk_energy_per_atom,
                "final_bulk": final_bulk,
            }
        )

    def calc(
        self,
        structure: dict[str, Any],  # type: ignore[override]
    ) -> dict[str, Any]:
        """
        Compute grain boundary energy for a single grain boundary structure dict produced by
        calc_gb or direct input. The function handles relaxation of both bulk and grain boundary
        structures when necessary.

        Args:
            structure: Dictionary containing information about the bulk and grain boundary
                structures. It must have the format:
                ``{'grain_boundary': gb_structure, 'bulk': bulk_structure}``
                or
                ``{'grain_boundary': gb_structure, 'bulk_energy_per_atom': energy}``.

        Returns:
            A dictionary containing the updated structure data, including fields like
            'grain_boundary', 'final_grain_boundary', 'gb_relax_energy',
            'grain_boundary_energy', and possibly updated 'bulk_energy_per_atom' and 'final_bulk'.

        Raises:
            ValueError: If structure is not a dict with the required keys.
        """
        if not (
            isinstance(structure, dict)
            and "grain_boundary" in structure
            and set(structure.keys()).intersection(("bulk", "bulk_energy_per_atom"))
        ):
            raise ValueError(
                "For grain boundary calculations, structure must be a dict in one of the following formats: "
                "{'grain_boundary': gb_structure, 'bulk': bulk_structure} or "
                "{'grain_boundary': gb_structure, 'bulk_energy_per_atom': energy}."
            )

        result_dict = structure.copy()

        if "bulk_energy_per_atom" in structure:
            bulk_energy_per_atom = structure["bulk_energy_per_atom"]
        else:
            logger.info("Relaxing bulk structure with both atomic and cell degrees of freedom")
            relaxer = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_cell=self.relax_bulk,
                relax_atoms=self.relax_bulk,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )
            bulk_opt = relaxer.calc(structure["bulk"])
            final_bulk = bulk_opt["final_structure"]
            bulk_energy_per_atom = bulk_opt["energy"] / len(final_bulk)
            result_dict["bulk_energy_per_atom"] = bulk_energy_per_atom
            result_dict["final_bulk"] = final_bulk

        # Relax GB
        gb_struct = structure["grain_boundary"]
        relax_gb_calc = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=self.relax_gb,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        gb_opt = relax_gb_calc.calc(gb_struct)
        final_gb = gb_opt["final_structure"]
        gb_energy = gb_opt["energy"]

        # Compute area (a * b)
        lattice = final_gb.lattice.matrix
        area = np.linalg.norm(np.cross(lattice[0], lattice[1]))
        n_atoms = len(final_gb)

        # Compute grain boundary energy
        gamma_gb = (gb_energy - n_atoms * bulk_energy_per_atom) / (2 * area)

        return result_dict | {
            "final_grain_boundary": final_gb,
            "gb_relax_energy": gb_energy,
            "grain_boundary_energy": gamma_gb,
        }
