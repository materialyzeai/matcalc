"""Adsorption energy calculations on surfaces."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import Slab, SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend._ase import get_ase_optimizer
from .utils import to_ase_atoms, to_pmg_molecule, to_pmg_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Molecule, Structure
    from pymatgen.core.surface import Slab


class AdsorptionCalc(PropCalc):
    """
    Adsorption energies on surfaces via bulk, slab, adsorbate, and adslab relaxations.

    Uses ``RelaxCalc`` for geometry optimizations. Attributes mirror constructor arguments
    plus optional cached bulk state after ``calc_bulk`` / ``calc_adslabs``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_adsorbate: bool = True,
        relax_slab: bool = True,
        relax_bulk: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            relax_adsorbate: Relax isolated adsorbate in a box when True.
            relax_slab: Relax clean slab (fixed cell) when True.
            relax_bulk: Relax bulk with cell when True.
            fmax: Force tolerance (eV/Å).
            optimizer: ASE optimizer name or class.
            max_steps: Max relaxation steps.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.relax_adsorbate = relax_adsorbate
        self.relax_bulk = relax_bulk
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

        self.final_bulk: Structure | None = None
        self.bulk_energy: float | None = None
        self.n_bulk_atoms: int | None = None

    def calc_bulk(
        self,
        bulk: Structure | Atoms,
        bulk_energy: float | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            bulk: Bulk pymatgen structure or ASE atoms.
            bulk_energy: If ``relax_bulk`` is False and this is set, used as energy; else relaxed.

        Returns:
            Dict with ``final_structure`` and ``energy`` (when available).
        """
        initial_bulk = to_pmg_structure(bulk)

        if self.relax_bulk:
            relaxer_bulk = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_cell=self.relax_bulk,
                relax_atoms=self.relax_bulk,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )

            bulk_opt = relaxer_bulk.calc(initial_bulk)
        else:
            bulk_opt = {
                "final_structure": bulk,
                "energy": bulk_energy if bulk_energy is not None else None,
            }

        return bulk_opt

    def calc_adsorbate(
        self,
        adsorbate: Molecule | Atoms,
        adsorbate_energy: float | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            adsorbate: Adsorbate as molecule or atoms.
            adsorbate_energy: Optional fixed energy; otherwise from calculator on final geometry.

        Returns:
            Dict with ``adsorbate_energy``, ``adsorbate``, ``final_adsorbate``.
        """
        initial_adsorbate = to_pmg_molecule(adsorbate)

        if self.relax_adsorbate:
            relaxer = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_atoms=self.relax_adsorbate,
                relax_cell=False,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )

            adsorbate = to_ase_atoms(adsorbate)
            # Add 15 Å of vacuum in all directions for relaxation
            adsorbate.set_cell(np.max(adsorbate.positions, axis=0) - np.min(adsorbate.positions, axis=0) + 15)
            adsorbate_opt = relaxer.calc(adsorbate)
        else:
            adsorbate_opt = {
                "final_structure": adsorbate,
            }
        if adsorbate_energy is not None:
            final_adsorbate_energy = adsorbate_energy
        else:
            final_adsorbate_energy = self.calculator.get_potential_energy(
                to_ase_atoms(adsorbate_opt["final_structure"])
            )

        final_adsorbate = to_pmg_molecule(adsorbate_opt["final_structure"])

        return {
            "adsorbate_energy": final_adsorbate_energy,
            "adsorbate": initial_adsorbate,
            "final_adsorbate": final_adsorbate,
        }

    def calc_slab(
        self,
        slab: Structure | Atoms,
        slab_energy: float | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            slab: Slab structure or atoms.
            slab_energy: Optional total slab energy; if None, taken from calculator.

        Returns:
            Dict with ``slab``, ``slab_energy``, ``slab_energy_per_atom``, ``final_slab``.
        """
        if self.relax_slab:
            relaxer = RelaxCalc(
                calculator=self.calculator,
                fmax=self.fmax,
                max_steps=self.max_steps,
                relax_atoms=self.relax_slab,
                relax_cell=False,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
            )
            slab_opt = relaxer.calc(slab)
        else:
            slab_opt = {
                "final_structure": slab,
            }

        if slab_energy is None:
            slab_energy = self.calculator.get_potential_energy(to_ase_atoms(slab))

        slab_energy_per_atom = slab_energy / len(slab) if slab_energy is not None else None

        return {
            "slab": slab,
            "slab_energy": slab_energy,
            "slab_energy_per_atom": slab_energy_per_atom,
            "final_slab": slab_opt["final_structure"],
        }

    def calc_adslabs(  # noqa: C901
        self,
        adsorbate: Molecule | Atoms,
        bulk: Structure | Atoms,
        miller_index: tuple[int, int, int],
        *,
        adsorbate_energy: float | None = None,
        slab_energy: float | None = None,
        min_slab_size: float = 10.0,
        min_vacuum_size: float = 20.0,
        min_area_extent: tuple[float, float] | None = None,
        inplane_supercell: tuple[int, int] = (1, 1),
        slab_gen_kwargs: dict[str, Any] | None = None,
        get_slabs_kwargs: dict[str, Any] | None = None,
        adsorption_sites: dict[str, Sequence[tuple[float, float]]] | str = "all",
        height: float = 0.9,
        mi_vec: tuple[float, float] | None = None,
        fixed_height: float = 5,
        find_adsorption_sites_args: dict[str, Any] | None = None,
        dry_run: bool = False,
        **kwargs: dict[str, Any],
    ) -> list[dict[str, Any]] | dict[Any, Any]:
        """
        Args:
            adsorbate: Adsorbate species.
            bulk: Bulk structure (use conventional cell if needed for Miller indices).
            miller_index: Surface orientation.
            adsorbate_energy: Optional fixed adsorbate energy.
            slab_energy: Optional fixed clean-slab energy for all slabs.
            min_slab_size: Minimum slab thickness (Å).
            min_vacuum_size: Vacuum thickness (Å).
            min_area_extent: Minimum in-plane extent (Å); may expand ``inplane_supercell``.
            inplane_supercell: In-plane supercell factors.
            slab_gen_kwargs: Kwargs for ``SlabGenerator``.
            get_slabs_kwargs: Kwargs for ``SlabGenerator.get_slabs``.
            adsorption_sites: ``"all"``, site name string, or custom fractional-coordinate dict.
            height: Adsorbate height above surface (Å).
            mi_vec: Optional in-plane vector for ``AdsorbateSiteFinder``.
            fixed_height: Fix slab atoms below this height (Å) during relaxation.
            find_adsorption_sites_args: Kwargs for ``find_adsorption_sites``.
            dry_run: If True, return adslab dicts without ``calc_many``.
            **kwargs: Forwarded to ``calc_many``.

        Returns:
            List of per-configuration results, or raw adslab dicts if ``dry_run``.
        """
        adslab_dict = {}
        bulk = to_pmg_structure(bulk)
        bulk_opt = self.calc_bulk(bulk)

        adsorbate_dict = self.calc_adsorbate(adsorbate, adsorbate_energy=adsorbate_energy)

        # Generally want the surface perpendicular to z
        if slab_gen_kwargs is not None:
            slab_gen_kwargs["max_normal_search"] = slab_gen_kwargs.get("max_normal_search", np.max(miller_index) + 1)
            slab_gen_kwargs["reorient_lattice"] = slab_gen_kwargs.get("reorient_lattice", True)

        slabgen = SlabGenerator(
            initial_structure=bulk_opt["final_structure"],
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            **(slab_gen_kwargs or {}),
        )

        slab_dicts = []
        for slab_ in slabgen.get_slabs(**(get_slabs_kwargs or {})):
            if min_area_extent is not None:
                avec = slab_.lattice.matrix[0]
                bvec = slab_.lattice.matrix[1]
                aperp = bvec - np.dot(bvec, avec) / np.dot(avec, avec) * avec
                na = int(np.ceil(min_area_extent[0] / np.linalg.norm(avec)))
                nb = int(np.ceil(min_area_extent[1] / np.linalg.norm(aperp)))
                inplane_supercell = (na, nb)

            slab_dicts.append(
                {
                    "slab": slab_.make_supercell((*inplane_supercell, 1), in_place=False),
                    "miller_index": miller_index,
                    "shift": slab_.shift,
                }
            )

        adslabs: list[dict[str, Any]] = []
        for slab_dict_ in slab_dicts:
            slab_dict = copy(slab_dict_)
            slab: Slab = cast("Slab", slab_dict["slab"])

            if fixed_height is not None:
                maxz = np.min(slab.cart_coords, axis=0)[2] + fixed_height
                fix_idx = np.argwhere(slab.cart_coords[:, 2] < maxz).flatten()
                slab.add_site_property(
                    "selective_dynamics",
                    [[False] * 3 if i in fix_idx else [True] * 3 for i in range(len(slab))],
                )

            slab_dict |= self.calc_slab(slab, slab_energy=slab_energy)
            slab_dict |= copy(adsorbate_dict)

            final_slab = cast("Slab", slab_dict["final_slab"])
            asf = AdsorbateSiteFinder(
                final_slab,
                height=height,
                mi_vec=mi_vec,
            )

            if adsorption_sites == "all":
                asf_adsites = asf.find_adsorption_sites(**find_adsorption_sites_args or {})
                asf_adsites.pop("all")
                adsites = {s: asf_adsites[s] for s in asf_adsites}
            elif isinstance(adsorption_sites, str):
                asf_adsites = asf.find_adsorption_sites(**find_adsorption_sites_args or {})
                try:
                    adsites = {adsorption_sites: asf_adsites[adsorption_sites]}
                except KeyError as err:
                    raise KeyError(
                        f"Provided sites: '{adsorption_sites}' must be one"
                        f" of {asf_adsites.keys()} or dictionary of the "
                        "form {'site_name': [(x1, y1, z1), (x2, y2, z2), ...]}."
                    ) from err
            else:
                adsites = adsorption_sites

            for adsite in adsites:
                for adsite_idx, adsite_coord in enumerate(adsites[adsite]):
                    adslab = asf.add_adsorbate(
                        molecule=adsorbate_dict["final_adsorbate"],
                        ads_coord=adsite_coord,
                    )
                    adslab_dict = {
                        "adslab": adslab,
                        "adsorption_site": adsite,
                        "adsorption_site_coord": adsite_coord,
                        "adsorption_site_index": adsite_idx,
                    }
                    adslab_dict |= copy(slab_dict)
                    adslabs.append(adslab_dict)

        if dry_run:
            return adslabs
        return list(self.calc_many(adslabs, **kwargs))  # type:ignore[arg-type]

    def calc(
        self,
        structure: dict[str, Any],  # type: ignore[override]
    ) -> dict[str, Any]:
        """
        Args:
            structure: Dict with ``adslab``, ``slab``, ``adsorbate``; optional ``slab_energy`` /
                ``adsorbate_energy``.

        Returns:
            Dict including ``adsorption_energy`` and merged slab/adsorbate fields.
        """
        if not isinstance(structure, dict):
            msg = (
                "For adsorption calculations, structure must be a dict with keys "
                "'adslab', 'slab', and 'adsorbate' (and optionally 'slab_energy_per_atom' "
                "and/or 'adsorbate_energy'). Use calc_adslabs(...) to generate input dicts."
            )
            raise TypeError(msg)
        result_dict = structure.copy()

        result_dict |= self.calc_adsorbate(
            structure["adsorbate"],
            adsorbate_energy=structure.get("adsorbate_energy"),
        )

        result_dict |= self.calc_slab(structure["slab"], slab_energy=structure.get("slab_energy"))

        try:
            adslab = structure["adslab"]
        except KeyError as err:
            raise ValueError(
                "For adsorption calculations, structure must be dict with"
                " keys ('adslab' and 'slab' and 'adsorbate') and optionally"
                " ('slab_energy_per_atom') and/or"
                " 'adsorbate_energy'"
            ) from err

        n_adsorbate_atoms = len(structure["adsorbate"])
        n_slab_atoms = len(adslab) - n_adsorbate_atoms

        adslab_atoms = to_ase_atoms(adslab)
        adslab_atoms.calc = self.calculator
        opt = get_ase_optimizer(self.optimizer)(adslab_atoms)  # type:ignore[operator]
        opt.run(fmax=self.fmax, steps=self.max_steps)
        final_adslab = to_pmg_structure(adslab_atoms)
        adslab_energy = adslab_atoms.get_potential_energy()
        ads_energy = (
            adslab_energy - n_slab_atoms * result_dict["slab_energy_per_atom"] - result_dict["adsorbate_energy"]
        )

        return result_dict | {
            "final_adslab": final_adslab,
            "adslab_energy": adslab_energy,
            "adsorption_energy": ads_energy,
        }
