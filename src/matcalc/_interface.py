"""Interface structure/energy calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.structure_matcher import StructureMatcher

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
    given their miller indices, relaxes them, and computes their interfacial energies.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        relax_bulk: bool = True,
        relax_interface: bool = True,
        fmax: float = 0.1,
        optimizer: str | Optimizer = "BFGS",
        max_steps: int = 500,
        relax_calc_kwargs: dict | None = None,
    ) -> None:
        """Initialize the instance of the class.

        Parameters:
            calculator (Calculator | str): An ASE calculator object used to perform energy and force
                calculations. If string is provided, the corresponding universal calculator is loaded.
            relax_bulk (bool, optional): Whether to relax the bulk structures before interface
                calculations. Defaults to True.
            relax_interface (bool, optional): Whether to relax the interface structures. Defaults to True.
            fmax (float, optional): The maximum force tolerance for convergence. Defaults to 0.1.
            optimizer (str | Optimizer, optional): The optimization algorithm to use. Defaults to "BFGS".
            max_steps (int, optional): The maximum number of optimization steps. Defaults to 500.
            relax_calc_kwargs: Additional keyword arguments passed to the
            class:`RelaxCalc` constructor for both bulk and interface. Default is None.

        Returns:
            None
        """
        self.calculator = calculator
        self.relax_bulk = relax_bulk
        self.relax_interface = relax_interface
        self.fmax = fmax
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.relax_calc_kwargs = relax_calc_kwargs

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

        relaxer_bulk = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=self.relax_bulk,
            relax_atoms=self.relax_bulk,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        film_opt = relaxer_bulk.calc(film_bulk)
        substrate_opt = relaxer_bulk.calc(substrate_bulk)

        interfaces = [
            {
                "interface": interface,
                "num_atoms": len(interface),
                "film_energy_per_atom": film_opt["energy"] / len(film_bulk),
                "final_film": film_opt["final_structure"],
                "substrate_energy_per_atom": substrate_opt["energy"] / len(substrate_bulk),
                "final_substrate": substrate_opt["final_structure"],
            }
            for interface in unique_interfaces
        ]

        return [r for r in self.calc_many(interfaces, **kwargs) if r is not None]

    def calc(
        self,
        structure: Structure | Atoms | dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate the interfacial energy of the given interface structures and sort by the energy.

        Parameters:
            structure : A dictionary containing the film, substrate, interface structures

        Returns:
            dict:
                - "interface" (Structure): The initial interface structure.
                - "final_interface" (Structure): The relaxed interface structure.
                - "interface_energy_per_atom" (float): The final energy of the relaxed interface structure.
                - "num_atoms" (int): The number of atoms in the interface structure.
                - "interfacial_energy" (float): The calculated interfacial energy
        """
        if not isinstance(structure, dict):
            msg = (
                "For interface calculations, structure must be a dict in one of the following formats: "
                "{'interface': interface_struct, 'film_energy_per_atom': energy, ...} from calc_interfaces or "
                "{'interface': interface_struct, 'film_bulk': film_struct, 'substrate_bulk': substrate_struct}."
            )
            raise TypeError(msg)

        # Type narrowing: at this point, structure is guaranteed to be dict[str, Any]
        result_dict = structure.copy()

        if "film_energy_per_atom" in structure and "substrate_energy_per_atom" in structure:
            film_energy_per_atom = structure["film_energy_per_atom"]
            substrate_energy_per_atom = structure["substrate_energy_per_atom"]
            film_structure = structure["final_film"]
            substrate_structure = structure["final_substrate"]
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
            film_opt = relaxer.calc(structure["film_bulk"])
            film_energy_per_atom = film_opt["energy"] / len(film_opt["final_structure"])
            film_structure = film_opt["final_structure"]
            substrate_opt = relaxer.calc(structure["substrate_bulk"])
            substrate_energy_per_atom = substrate_opt["energy"] / len(substrate_opt["final_structure"])
            substrate_structure = substrate_opt["final_structure"]

        interface = structure["interface"]
        relaxer = RelaxCalc(
            calculator=self.calculator,
            fmax=self.fmax,
            max_steps=self.max_steps,
            relax_cell=False,
            relax_atoms=self.relax_interface,
            optimizer=self.optimizer,
            **(self.relax_calc_kwargs or {}),
        )
        interface_opt = relaxer.calc(interface)
        final_interface = interface_opt["final_structure"]
        interface_energy = interface_opt["energy"]

        # pymatgen interface object does not include interface properties for interfacial energy
        # calculation, define them here

        matrix = interface.lattice.matrix
        area = float(np.linalg.norm(np.cross(matrix[0], matrix[1])))

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
            msg = "No unique elements found in either structure to determine atom counts in interface."
            raise ValueError(msg)

        gamma = (
            interface_energy
            - (film_in_interface * film_energy_per_atom + substrate_in_interface * substrate_energy_per_atom)
        ) / (2 * area)

        return result_dict | {
            "interface": interface,
            "final_interface": final_interface,
            "interface_energy_per_atom": interface_energy / len(interface),
            "num_atoms": len(interface),
            "interfacial_energy": gamma,
        }
