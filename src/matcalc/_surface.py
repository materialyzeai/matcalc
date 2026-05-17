"""Surface Energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatgen.core.surface import SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class SurfaceCalc(PropCalc):
    """
    A class for performing surface energy calculations by generating and optionally
    relaxing bulk and slab structures. This facilitates materials science and
    computational chemistry workflows, enabling computations of surface properties
    for various crystal orientations and surface terminations.

    Attributes:
        calculator: ASE calculator or universal model name.
        relax_bulk: Relax bulk with cell when True.
        relax_slab: Relax slab (fixed cell) when True.
        fmax: Force tolerance (eV/Å).
        optimizer: ASE optimizer name or class.
        max_steps: Max relaxation steps.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        final_bulk: Relaxed bulk after ``calc_slabs`` (if run).
        bulk_energy: Total energy of relaxed bulk.
        n_bulk_atoms: Atom count in relaxed bulk.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_bulk: bool = True,
        relax_slab: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            relax_bulk: Relax bulk and cell before slab generation.
            relax_slab: Relax each slab (fixed cell).
            fmax: Force convergence (eV/Å).
            optimizer: ASE optimizer name or class.
            max_steps: Maximum relaxation steps.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.relax_bulk = relax_bulk
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

        self.final_bulk: Structure | None = None
        self.bulk_energy: float | None = None
        self.n_bulk_atoms: int | None = None

    def calc_slabs(
        self,
        bulk: Structure,
        miller_index: tuple[int, int, int] = (1, 0, 0),
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
        symmetrize: bool = True,  # noqa: FBT001,FBT002
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict | None = None,
        get_slabs_kwargs: dict | None = None,
        **kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Calculates slabs based on a given bulk structure and generates a set of slabs
        using specified parameters. The function leverages slab generation tools and
        defines the in-plane supercell, symmetry, and optimizes the bulk structure
        prior to slab generation. This is useful for surface calculations in materials
        science and computational chemistry.

        Args:
            bulk: Pymatgen bulk structure (converted to conventional cell).
            miller_index: Slab orientation.
            min_slab_size: Minimum slab thickness (Å).
            min_vacuum_size: Vacuum thickness (Å).
            symmetrize: Symmetrize slabs in ``SlabGenerator.get_slabs``.
            inplane_supercell: In-plane supercell factors for each slab.
            slab_gen_kwargs: Extra kwargs for ``SlabGenerator``.
            get_slabs_kwargs: Extra kwargs for ``get_slabs``.
            **kwargs: Forwarded to ``calc_many``.

        Returns:
            List of per-slab result dicts from ``calc_many``.
        """
        bulk = bulk.to_conventional()

        relaxer_bulk = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=self.relax_bulk,
            relax_atoms=self.relax_bulk,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        bulk_opt = relaxer_bulk.calc(bulk)

        slabgen = SlabGenerator(
            initial_structure=bulk,
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            **(slab_gen_kwargs or {}),
        )
        slabs = [
            {
                "slab": slab.make_supercell((*inplane_supercell, 1)),
                "bulk_energy_per_atom": bulk_opt["energy"] / len(bulk),
                "final_bulk": bulk_opt["final_structure"],
                "miller_index": miller_index,
            }
            for slab in slabgen.get_slabs(symmetrize=symmetrize, **(get_slabs_kwargs or {}))
        ]

        return list(self.calc_many(slabs, **kwargs))  # type:ignore[arg-type]

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Performs surface energy calculation for a given structure dictionary. The function handles
        the relaxation of both bulk and slab structures when necessary and computes the surface energy
        using the slab's relaxed energy, number of atoms, bulk energy per atom, and surface area.

        Args:
            structure: Dict with ``slab`` and either ``bulk`` or ``bulk_energy_per_atom``,
                or a structure type (invalid combinations raise).

        Returns:
            Dict with ``final_slab``, ``slab_energy``, ``surface_energy``, and related bulk fields.
        """
        if not (isinstance(structure, dict) and set(structure.keys()).intersection(("bulk", "bulk_energy_per_atom"))):
            raise ValueError(
                "For surface calculations, structure must be a dict in one of the following formats: "
                "{'slab': slab_struct, 'bulk': bulk_struct} or {'slab': slab_struct, 'bulk_energy': energy}."
            )

        result_dict = structure.copy()

        if "bulk_energy_per_atom" in structure:
            bulk_energy_per_atom = structure["bulk_energy_per_atom"]
        else:
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
            bulk_energy_per_atom = bulk_opt["energy"] / len(bulk_opt["final_structure"])
            result_dict["bulk_energy_per_atom"] = bulk_energy_per_atom
            result_dict["final_bulk"] = bulk_opt["final_structure"]

        slab = structure["slab"]
        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=self.relax_slab,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        slab_opt = relaxer.calc(slab)
        final_slab = slab_opt["final_structure"]
        slab_energy = slab_opt["energy"]

        # Compute surface energy
        # Assuming two surfaces: (E_slab - N_slab_atoms * E_bulk_per_atom) / (2 * area)
        area = slab.surface_area
        n_slab_atoms = len(final_slab)
        gamma = (slab_energy - n_slab_atoms * bulk_energy_per_atom) / (2 * area)

        return result_dict | {
            "slab": slab,
            "final_slab": final_slab,
            "slab_energy": slab_energy,
            "surface_energy": gamma,
        }
