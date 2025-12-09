"""Calculator for phonon properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class PhononCalc(PropCalc):
    """
    A class for phonon and thermal property calculations using phonopy.

    The `PhononCalc` class extends the `PropCalc` class to provide
    functionalities for calculating phonon properties and thermal properties
    of a given structure using the phonopy library. It includes options for
    structural relaxation before phonon property determination, as well as
    methods to export calculated properties to various output files for
    further analysis or visualization.

    :ivar calculator: A calculator object or a string specifying the
        computational backend to be used.
    :type calculator: Calculator | str
    :ivar atom_disp: Magnitude of atomic displacements for phonon
        calculations.
    :type atom_disp: float
    :ivar supercell_matrix: Array defining the transformation matrix to
        construct supercells for phonon calculations.
    :type supercell_matrix: ArrayLike
    :ivar t_step: Temperature step for thermal property calculations in
        Kelvin.
    :type t_step: float
    :ivar t_max: Maximum temperature for thermal property calculations in
        Kelvin.
    :type t_max: float
    :ivar t_min: Minimum temperature for thermal property calculations in
        Kelvin.
    :type t_min: float
    :ivar fmax: Maximum force convergence criterion for structural relaxation.
    :type fmax: float
    :ivar optimizer: String specifying the optimizer type to be used for
        structural relaxation.
    :type optimizer: str
    :ivar relax_structure: Boolean flag to determine whether to relax the
        structure before phonon calculation.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Optional dictionary containing additional
        arguments for the structural relaxation calculation.
    :type relax_calc_kwargs: dict | None
    :ivar write_force_constants: Path, boolean, or string specifying whether
        to write the calculated force constants to an output file, and the
        path or name of the file if applicable.
    :type write_force_constants: bool | str | Path
    :ivar write_band_structure: Path, boolean, or string specifying whether
        to write the calculated phonon band structure to an output file,
        and the path or name of the file if applicable.
    :type write_band_structure: bool | str | Path
    :ivar write_total_dos: Path, boolean, or string specifying whether to
        write the calculated total density of states (DOS) to an output
        file, and the path or name of the file if applicable.
    :type write_total_dos: bool | str | Path
    :ivar write_phonon: Path, boolean, or string specifying whether to write
        the calculated phonon properties (e.g., phonon.yaml) to an output
        file, and the path or name of the file if applicable.
    :type write_phonon: bool | str | Path
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        write_force_constants: bool | str | Path = False,
        write_band_structure: bool | str | Path = False,
        write_total_dos: bool | str | Path = False,
        write_phonon: bool | str | Path = True,
    ) -> None:
        """
        Initializes the class with configuration for the phonon calculations. The initialization parameters control
        the behavior of structural relaxation, thermal properties, force calculations, and output file generation.
        The class allows for customization of key parameters to facilitate the study of material behaviors.

        :param calculator: The calculator object or string name specifying the calculation backend to use.
        :param atom_disp: Atom displacement to be used for finite difference calculation of force constants.
        :param supercell_matrix: Transformation matrix to define the supercell for the calculation.
        :param t_step: Temperature step for thermal property calculations.
        :param t_max: Maximum temperature for thermal property calculations.
        :param t_min: Minimum temperature for thermal property calculations.
        :param fmax: Maximum force during structure relaxation, used as a convergence criterion.
        :param optimizer: Name of the optimization algorithm for structural relaxation.
        :param relax_structure: Flag to indicate whether structure relaxation should be performed before calculations.
        :param relax_calc_kwargs: Additional keyword arguments for relaxation phase calculations.
        :param write_force_constants: File path or boolean flag to write force constants.
            Defaults to "force_constants".
        :param write_band_structure: File path or boolean flag to write band structure data.
            Defaults to "band_structure.yaml".
        :param write_total_dos: File path or boolean flag to write total density of states (DOS) data.
            Defaults to "total_dos.dat".
        :param write_phonon: File path or boolean flag to write phonon data. Defaults to "phonon.yaml".
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.supercell_matrix = supercell_matrix
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.write_force_constants = write_force_constants
        self.write_band_structure = write_band_structure
        self.write_total_dos = write_total_dos
        self.write_phonon = write_phonon

        # Set default paths for output files.
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            phonon: Phonopy object with force constants produced
            thermal_properties:
                {
                    temperatures: list of temperatures in Kelvin,
                    free_energy: list of Helmholtz free energies at corresponding temperatures in kJ/mol,
                    entropy: list of entropies at corresponding temperatures in J/K/mol,
                    heat_capacity: list of heat capacities at constant volume at corresponding temperatures in J/K/mol,
                    The units are originally documented in phonopy.
                    See phonopy.Phonopy.run_thermal_properties()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/api_phonopy.py#L2591)
                    -> phonopy.phonon.thermal_properties.ThermalProperties.run()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L498)
                    -> phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_free_energy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L217)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_entropy()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L233)
                    phonopy.phonon.thermal_properties.ThermalPropertiesBase.run_heat_capacity()
                    (https://github.com/phonopy/phonopy/blob/develop/phonopy/phonon/thermal_properties.py#L225)
                }
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]
        cell = get_phonopy_structure(to_pmg_structure(structure_in))
        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = phonon.supercells_with_displacements
        phonon.forces = [  # type: ignore[assignment]
            run_pes_calc(get_pmg_structure(supercell), self.calculator).forces
            for supercell in disp_supercells  # type:ignore[union-attr]
            if supercell is not None
        ]
        phonon.produce_force_constants()
        phonon.run_mesh()
        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)  # type: ignore[arg-type]
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=self.write_band_structure)
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)  # type: ignore[arg-type]
        return result | {"phonon": phonon, "thermal_properties": phonon.get_thermal_properties_dict()}
