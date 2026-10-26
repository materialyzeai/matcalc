"""Interface structure/energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.core.structure_matcher import StructureMatcher

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator
    from pymatgen.core import Structure


class InterfaceCalc(PropCalc):
    """
    This class generates all possible coherent interfaces between two bulk structures
    given their miller indices, relaxes them, and computes their interfacial and/or adhesion energies.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        interfacial_energy: bool = True,
        adhesion_energy: bool = False,
        relax_bulk: bool = True,
        relax_interface: bool = True,
        relax_slab: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """Initialize the instance of the class.

        Parameters:
            calculator (Calculator | str): An ASE calculator object used to perform energy and force
                calculations. If string is provided, the corresponding universal calculator is loaded.
            interfacial_energy (bool, optional): Whether to compute the interfacial energy. Defaults to True.
            adhesion_energy (bool, optional): Whether to compute the adhesion energy. Defaults to False.
            relax_bulk (bool, optional): Whether to relax the bulk structures before interface
                calculations. Defaults to True.
            relax_interface (bool, optional): Whether to relax the interface structures. Defaults to True.
            relax_slab (bool, optional): Whether to relax the atoms of the film and substrate slabs at fixed
                cell. Defaults to True. For fixed geometries, set relax_interface and relax_slab to False.
            fmax (float, optional): The maximum force tolerance for convergence. Defaults to 0.1.
            optimizer (str | Optimizer, optional): The optimization algorithm to use. Defaults to "BFGS".
            max_steps (int, optional): The maximum number of optimization steps. Defaults to 500.
            relax_calc_kwargs: Additional keyword arguments passed to the
                :class:`RelaxCalc` constructor for the bulk, interface and slab relaxations. They override the
                settings above (e.g. relax_cell, fmax) for all of these relaxations. Default is None.

        Returns:
            None
        """
        self.calculator = calculator
        self.interfacial_energy = interfacial_energy
        self.adhesion_energy = adhesion_energy
        self.relax_bulk = relax_bulk
        self.relax_interface = relax_interface
        self.relax_slab = relax_slab
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

    def _relax_calc(self, *, relax_cell: bool, relax_atoms: bool) -> RelaxCalc:
        """RelaxCalc with the instance settings; entries in relax_calc_kwargs override them."""
        kwargs = {
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            "relax_cell": relax_cell,
            "relax_atoms": relax_atoms,
            "optimizer": self.optimizer,
            **(self.relax_calc_kwargs or {}),
        }
        return RelaxCalc(calculator=self.calculator, **kwargs)

    def calc_bulk(self, bulk: Structure | Atoms) -> dict[str, Any]:
        """Relax a bulk structure if relax_bulk, and compute its energy.

        Parameters:
            bulk (Structure | Atoms): The bulk structure.

        Returns:
            dict: "bulk", "bulk_energy" (eV), "bulk_energy_per_atom" (eV/atom) and "final_bulk".
        """
        bulk_opt = self._relax_calc(relax_cell=self.relax_bulk, relax_atoms=self.relax_bulk).calc(bulk)
        return {
            "bulk": bulk,
            "bulk_energy": bulk_opt["energy"],
            "bulk_energy_per_atom": bulk_opt["energy"] / len(bulk),
            "final_bulk": bulk_opt["final_structure"],
        }

    def calc_slab(self, slab: Structure | Atoms) -> dict[str, Any]:
        """Relax the atoms of a slab at fixed cell if relax_slab, and compute its energy.

        Parameters:
            slab (Structure | Atoms): The slab structure.

        Returns:
            dict: "slab", "slab_energy" (eV), "slab_energy_per_atom" (eV/atom) and "final_slab".
        """
        slab_opt = self._relax_calc(relax_cell=False, relax_atoms=self.relax_slab).calc(slab)
        return {
            "slab": slab,
            "slab_energy": slab_opt["energy"],
            "slab_energy_per_atom": slab_opt["energy"] / len(slab),
            "final_slab": slab_opt["final_structure"],
        }

    def calc_interfaces(
        self,
        film_bulk: Structure,
        substrate_bulk: Structure,
        film_miller: tuple[int, int, int],
        substrate_miller: tuple[int, int, int],
        zslgen: ZSLGenerator | None = None,
        cib_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Calculate all possible coherent interfaces between two bulk structures.

        Parameters:
            film_bulk (Structure): The bulk structure of the film material.
            substrate_bulk (Structure): The bulk structure of the substrate material.
            film_miller (tuple[int, int, int]): The Miller index for the film surface.
            substrate_miller (tuple[int, int, int]): The Miller index for the substrate surface.
            zslgen (ZSLGenerator | None, optional): An instance of ZSLGenerator to use for generating
                supercells.
            zsl_kwargs (dict | None, optional): Additional keyword arguments to pass to the
                ZSLGenerator.
            cib_kwargs (dict | None, optional): Additional keyword arguments to pass to the
                CoherentInterfaceBuilder.
            **kwargs (Any): Additional keyword arguments passed to calc_many.

        Returns:
                dict: A list of dictionaries containing the calculated film, substrate, interface.
        """
        cib = CoherentInterfaceBuilder(
            film_structure=film_bulk,
            substrate_structure=substrate_bulk,
            film_miller=film_miller,
            substrate_miller=substrate_miller,
            zslgen=zslgen,
            **(cib_kwargs or {}),
        )

        terminations = cib.terminations
        all_interfaces: list = []
        for t in terminations:
            interfaces_for_termination = cib.get_interfaces(termination=t)
            all_interfaces.extend(list(interfaces_for_termination))

        if not all_interfaces:
            raise ValueError(
                "No interfaces found with the given parameters. Adjust the ZSL parameters to find more matches."
            )

        # Group similar / duplicate interfaces using StructureMatcher and keep one representative per group
        matcher = StructureMatcher()
        groups: list[list] = []
        for i in all_interfaces:
            placed = False
            for g in groups:
                if matcher.fit(i, g[0]):
                    g.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
        unique_interfaces = [g[0] for g in groups]

        film_bulk = film_bulk.to_conventional()
        substrate_bulk = substrate_bulk.to_conventional()

        film = self.calc_bulk(film_bulk)
        substrate = self.calc_bulk(substrate_bulk)

        interfaces = [
            {
                "interface": interface,
                "num_atoms": len(interface),
                "film_energy_per_atom": film["bulk_energy_per_atom"],
                "final_film": film["final_bulk"],
                "substrate_energy_per_atom": substrate["bulk_energy_per_atom"],
                "final_substrate": substrate["final_bulk"],
            }
            for interface in unique_interfaces
        ]

        return [r for r in self.calc_many(interfaces, **kwargs) if r is not None]

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate the interfacial energy and/or the adhesion energy of the given interface structure.

        Both energies are in eV/A^2, with E_interface the energy of the final interface, N_film and N_substrate its
        numbers of film and substrate atoms, e_film and e_substrate the bulk energies per atom, and A = |a x b| of the
        interface cell:

            interfacial_energy = (E_interface - N_film * e_film - N_substrate * e_substrate) / (2 * A)
            adhesion_energy = (E_film_slab + E_substrate_slab - E_interface) / A

        The adhesion energy is positive for a bound interface and assumes one film/substrate contact per cell, as in
        the interfaces from calc_interfaces. The slabs are "film_slab" and "substrate_slab" if given, otherwise the
        film and substrate sites of the final interface in the same cell. With relax_slab, their atoms are relaxed at
        fixed cell; otherwise, slabs cut from the interface give the rigid work of separation.

        Parameters:
            structure : A dictionary with the "interface" structure. For the interfacial energy, it also contains
                "film_energy_per_atom", "substrate_energy_per_atom", "final_film" and "final_substrate" (from
                calc_interfaces), or "film_bulk" and "substrate_bulk". For the adhesion energy, it may contain
                "film_slab" and "substrate_slab" with the film and substrate atoms in the interface cell; otherwise,
                the interface needs the "interface_label" site property of a pymatgen Interface.

        Returns:
            dict:
                - "interface" (Structure): The initial interface structure.
                - "final_interface" (Structure): The relaxed interface structure.
                - "interface_energy_per_atom" (float): The final energy of the relaxed interface structure.
                - "num_atoms" (int): The number of atoms in the interface structure.
                - "interfacial_energy" (float): The calculated interfacial energy, if interfacial_energy.
                - "final_film_slab", "final_substrate_slab" (Structure), "film_slab_energy",
                  "substrate_slab_energy" (float, eV) and "adhesion_energy" (float), if adhesion_energy.
        """
        if not isinstance(structure, dict):
            msg = (
                "For interface calculations, structure must be a dict in one of the following formats: "
                "{'interface': interface_struct, 'film_energy_per_atom': energy, ...} from calc_interfaces or "
                "{'interface': interface_struct, 'film_bulk': film_struct, 'substrate_bulk': substrate_struct}, "
                "optionally with 'film_slab' and 'substrate_slab' for the adhesion energy."
            )
            raise TypeError(msg)

        has_slabs = [key in structure for key in ("film_slab", "substrate_slab")]
        if any(has_slabs) and not all(has_slabs):
            missing = "substrate_slab" if has_slabs[0] else "film_slab"
            msg = f"'film_slab' and 'substrate_slab' must be given together, but {missing!r} is missing."
            raise ValueError(msg)
        labels: list[str] = getattr(structure["interface"], "site_properties", {}).get("interface_label") or []
        if self.adhesion_energy and not all(has_slabs) and not labels:
            msg = (
                "The adhesion energy needs the film and substrate slabs: pass 'film_slab' and 'substrate_slab', "
                "or a pymatgen Interface (with the 'interface_label' site property) as 'interface'."
            )
            raise ValueError(msg)

        # Type narrowing: at this point, structure is guaranteed to be dict[str, Any]
        result_dict = structure.copy()

        interface = structure["interface"]
        bulk_energy: float | None = None
        if self.interfacial_energy:
            if "film_energy_per_atom" in structure and "substrate_energy_per_atom" in structure:
                film = {
                    "bulk_energy_per_atom": structure["film_energy_per_atom"],
                    "final_bulk": structure["final_film"],
                }
                substrate = {
                    "bulk_energy_per_atom": structure["substrate_energy_per_atom"],
                    "final_bulk": structure["final_substrate"],
                }
            else:
                film = self.calc_bulk(structure["film_bulk"])
                substrate = self.calc_bulk(structure["substrate_bulk"])
            film_in_interface, substrate_in_interface = self._count_atoms(
                interface, film["final_bulk"], substrate["final_bulk"]
            )
            bulk_energy = (
                film_in_interface * film["bulk_energy_per_atom"]
                + substrate_in_interface * substrate["bulk_energy_per_atom"]
            )
        interface_opt = self._relax_calc(relax_cell=False, relax_atoms=self.relax_interface).calc(interface)
        final_interface = interface_opt["final_structure"]
        interface_energy = interface_opt["energy"]

        # pymatgen interface object does not include interface properties for interfacial energy
        # calculation, define them here

        matrix = interface.lattice.matrix
        area = float(np.linalg.norm(np.cross(matrix[0], matrix[1])))

        result_dict |= {
            "interface": interface,
            "final_interface": final_interface,
            "interface_energy_per_atom": interface_energy / len(interface),
            "num_atoms": len(interface),
        }
        if bulk_energy is not None:
            result_dict["interfacial_energy"] = (interface_energy - bulk_energy) / (2 * area)
        if self.adhesion_energy:
            if all(has_slabs):
                film_slab, substrate_slab = structure["film_slab"], structure["substrate_slab"]
            else:
                # Labels are taken from the input interface; the fixed-cell relaxation keeps the site order.
                film_slab = final_interface.copy().remove_sites([i for i, t in enumerate(labels) if "substrate" in t])
                substrate_slab = final_interface.copy().remove_sites([i for i, t in enumerate(labels) if "film" in t])
            film = self.calc_slab(film_slab)
            substrate = self.calc_slab(substrate_slab)
            result_dict |= {
                "final_film_slab": film["final_slab"],
                "film_slab_energy": film["slab_energy"],
                "final_substrate_slab": substrate["final_slab"],
                "substrate_slab_energy": substrate["slab_energy"],
                "adhesion_energy": (film["slab_energy"] + substrate["slab_energy"] - interface_energy) / area,
            }
        return result_dict

    @staticmethod
    def _count_atoms(
        interface: Structure, film_structure: Structure, substrate_structure: Structure
    ) -> tuple[float, float]:
        """Count the film and substrate atoms in the interface."""
        # Prefer pymatgen's per-site film/substrate indices when available
        # (this is the only reliable signal for alloys / shared compositions).
        film_indices = getattr(interface, "film_indices", None)
        substrate_indices = getattr(interface, "substrate_indices", None)
        if film_indices is not None and substrate_indices is not None:
            film_in_interface = float(len(film_indices))
            substrate_in_interface = float(len(substrate_indices))
        else:
            unique_in_film = set(film_structure.symbol_set) - set(substrate_structure.symbol_set)
            unique_in_substrate = set(substrate_structure.symbol_set) - set(film_structure.symbol_set)

            if unique_in_film:
                unique_element = next(iter(unique_in_film))
                count = film_structure.composition[unique_element]
                film_in_interface = (interface.composition[unique_element] / count) * film_structure.num_sites
                substrate_in_interface = interface.num_sites - film_in_interface
            elif unique_in_substrate:
                unique_element = next(iter(unique_in_substrate))
                count = substrate_structure.composition[unique_element]
                substrate_in_interface = (interface.composition[unique_element] / count) * substrate_structure.num_sites
                film_in_interface = interface.num_sites - substrate_in_interface
            else:
                msg = (
                    "Could not determine film vs. substrate atom counts: the interface object "
                    "does not expose film_indices / substrate_indices and the two structures "
                    "share their full elemental set. Pass a pymatgen Interface with film_indices "
                    "set (built via CoherentInterfaceBuilder) or supply pre-computed energies "
                    "via film_energy_per_atom / substrate_energy_per_atom."
                )
                raise ValueError(msg)

        return film_in_interface, substrate_in_interface
