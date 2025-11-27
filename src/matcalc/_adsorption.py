"""Surface Energy calculations."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator

from ._base import PropCalc
from ._relaxation import RelaxCalc
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
    Calculator for adsorption energies on surfaces.
    Combines bulk relaxation, slab generation and relaxation, adsorbate
    relaxation, and adsorption energy calculations.
    Uses :class:`RelaxCalc` for structure relaxations.
    Parameters allow control over which parts are relaxed and how.

    :param calculator: An ASE calculator object used to perform energy and force
        calculations. If string is provided, the corresponding universal calculator is loaded.
    :type calculator: Calculator | str
    :param relax_adsorbate: Whether to relax the adsorbate structure. If note relaxed,
        a single point or the adsorbate energy provided downstream is used. Default is True.
    :type relax_adsorbate: bool, optional
    :param relax_slab: Whether to relax each clean slab structure (cell fixed). Default is True.
    :type relax_slab: bool, optional
    :param relax_bulk: Whether to relax the bulk structure used to generate slabs, including its cell. Default is True.
    :type relax_bulk: bool, optional
    :param fmax: Force tolerance in eV/Å for relaxation. Default is 0.1.
    :type fmax: float, optional
    :param optimizer: The ASE optimizer to usein RelaxCalc. Can be a string (e.g. "BFGS") or
        an :class:`Optimizer` instance. Default is "BFGS".
    :type optimizer: str | Optimizer, optional
    :param max_steps: Maximum number of optimization steps for relaxation. Default is 500.
    :type max_steps: int, optional
    :param relax_calc_kwargs: Additional keyword arguments passed to the
        :class:`RelaxCalc` constructor for both bulk and slabs. Default is None.
    :type relax_calc_kwargs: dict | None, optional
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
        Initialize the AdsorptionCalc.

        :param calculator: An ASE calculator object used to perform energy and force
            calculations. If string is provided, the corresponding universal calculator is loaded.
        :type calculator: Calculator | str
        :param relax_adsorbate: Whether to relax the adsorbate structure. Default is True.
        :type relax_adsorbate: bool, optional
        :param relax_slab: Whether to relax each slab structure (cell fixed). Default is True.
        :type relax_slab: bool, optional
        :param relax_bulk: Whether to relax the bulk structure, including its cell. Default is True.
        :type relax_bulk: bool, optional
        :param fmax: Force tolerance in eV/Å for relaxation. Default is 0.1.
        :type fmax: float, optional
        :param optimizer: The ASE optimizer to use. Can be a string (e.g. "BFGS") or
            an :class:`Optimizer` instance. Default is "BFGS".
        :type optimizer: str | Optimizer, optional
        :param max_steps: Maximum number of optimization steps for relaxation. Default is 500.
        :type max_steps: int, optional
        :param relax_calc_kwargs: Additional keyword arguments passed to the
            :class:`RelaxCalc` constructor for both bulk and slabs. Default is None.
        :type relax_calc_kwargs: dict | None, optional
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
        Prepare the bulk structure for slab generation, optionally relaxing it.

        :param bulk: The bulk structure to be calculated.
        :type bulk: Structure | Atoms
        :param bulk_energy: Optional pre-calculated energy of the bulk. If not provided,
            the bulk will be relaxed and its energy calculated.
        :type bulk_energy: float | None, optional
        :return: A dictionary containing the bulk energy and final structure.
        :rtype: dict[str, Any]
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
                "energy": bulk_energy
                if bulk_energy is not None else None,
            }

        return bulk_opt

    def calc_adsorbate(
        self,
        adsorbate: Molecule | Atoms,
        adsorbate_energy: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate the energy of the adsorbate, optionally relaxing it.

        :param adsorbate: The adsorbate structure to be calculated.
        :type adsorbate: Molecule | Atoms
        :return: A dictionary containing the adsorbate energy and final structure.
        :rtype: dict[str, Any]
        """
        initial_adsorbate = to_pmg_molecule(adsorbate)

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
        final_adsorbate = to_pmg_molecule(adsorbate_opt["final_structure"])
        final_adsorbate_energy = adsorbate_opt["energy"]

        if adsorbate_energy is not None:
            final_adsorbate_energy = adsorbate_energy

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
        Calculate the energy of the slab, optionally relaxing it.

        :param slab: The slab structure to be calculated.
        :type slab: Structure | Atoms
        :param slab_energy: Optional pre-calculated energy of the slab. If not provided,
            the slab will be relaxed and its energy calculated.
        :type slab_energy: float | None, optional
        :return: A dictionary containing the slab energy and final structure.
        :rtype: dict[str, Any]
        """
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
        if slab_energy is not None:
            slab_opt["energy"] = slab_energy

        return {
            "slab": slab,
            "slab_energy": slab_opt["energy"],
            "slab_energy_per_atom": slab_opt["energy"] / len(slab_opt["final_structure"]),
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
        Calculate adsorption energies for adsorbates on slabs generated from a bulk structure.
        :param adsorbate: The adsorbate structure to be placed on the slab.
        :type adsorbate: Molecule | Atoms
        :param bulk: The bulk structure from which slabs will be generated. Be careful
            to provide conventional cells if that is your intention for miller indices.
        :type bulk: Structure | Atoms
        :param miller_index: The Miller index defining the slab orientation.
        :type miller_index: tuple[int, int, int]
        :param adsorbate_energy: Optional pre-calculated energy of the adsorbate. If not provided,
            the adsorbate will be relaxed and its energy calculated.
        :type adsorbate_energy: float | None, optional
        :param min_slab_size: Minimum thickness of the slab in Å. Default is 10.0.
        :type min_slab_size: float, optional
        :param min_vacuum_size: Minimum size of the vacuum layer in Å. Default is 20.0.
        :type min_vacuum_size: float, optional
        :param min_area_extent: Minimum in-plane dimensions of the slab in Å. If provided,
            the slab will be expanded to at least these dimensions in the a and perpendicular
            to a directions by projecting the b lattice vector onto a. Default is None.
        :type min_area_extent: tuple[float, float] | None, optional
        :param inplane_supercell: Tuple defining the in-plane supercell size. Default is (1, 1).
        :type inplane_supercell: tuple[int, int], optional
        :param slab_gen_kwargs: Additional keyword arguments passed to the SlabGenerator. Default is None.
        :type slab_gen_kwargs: dict[str, Any] | None, optional
        :param get_slabs_kwargs: Additional keyword arguments passed to the get_slabs method of
            SlabGenerator. Default is None.
        :type get_slabs_kwargs: dict[str, Any] | None, optional
        :param adsorption_sites: Either a string specifying which adsorption sites to consider
            (e.g., "all", "ontop", "bridge", "hollow"), or a dictionary specifying custom adsorption sites
            with site names as keys and lists of fractional coordinates as values.
            Default is "all".
        :type adsorption_sites: dict[str:tuple[float, float]] | str, optional
        :param height: Height above the surface to place the adsorbate in Å. Default is 0.9.
        :type height: float, optional
        :param mi_vec: Optional in-plane vector defining the slab orientation. If None, it is
            automatically determined. Default is None.
        :type mi_vec: tuple[float, float] | None, optional
        :param fixed_height: Height below which slab atoms are fixed during relaxation in Å.
            Default is 5 Å.
        :type fixed_height: float, optional
        :param find_adsorption_sites_args: Additional keyword arguments passed to the
            find_adsorption_sites method of AdsorbateSiteFinder. Default is None.
        :type find_adsorption_sites_args: dict[str, Any] | None, optional
        :param dry_run: If True, only generates the adslab structures without performing calculations.
            Default is False.
        :type dry_run: bool, optional
        :return: A list of dictionaries containing results for each adslab, or just the structures if dry_run is True.
        :rtype: list[dict[str, Any]] | dict[Any, Any].
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
        Calculate the adsorption energy for a given adslab structure.

        :param structure: A dictionary containing 'adslab', 'slab', and 'adsorbate' structures,
            and optionally 'slab_energy' and/or 'adsorbate_energy'.
        :type structure: dict[str, Any]
        :return: A dictionary containing the adsorption energy and related information.
        :rtype: dict[str, Any]
        """
        result_dict = structure.copy()

        result_dict |= self.calc_adsorbate(
            structure["adsorbate"],
            adsorbate_energy=structure.get("adsorbate_energy"),
        )

        result_dict |= self.calc_slab(
            structure["slab"], slab_energy=structure.get("slab_energy")
        )

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
        if len(adslab) != n_slab_atoms + n_adsorbate_atoms:
            raise ValueError(
                "The number of atoms in the adslab does not equal the sum of the slab and adsorbate atoms."
            )

        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=True,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        adslab_opt = relaxer.calc(adslab)
        final_adslab = adslab_opt["final_structure"]
        adslab_energy = adslab_opt["energy"]

        ads_energy = (
            adslab_energy - n_slab_atoms * result_dict["slab_energy_per_atom"] - result_dict["adsorbate_energy"]
        )

        return result_dict | {
            "final_adslab": final_adslab,
            "adslab_energy": adslab_energy,
            "adsorption_energy": ads_energy,
        }
