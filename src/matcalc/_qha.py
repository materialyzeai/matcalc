"""Calculator for phonon properties under quasi-harmonic approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from phonopy import PhonopyQHA

from ._base import PropCalc
from ._phonon import PhononCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


class QHACalc(PropCalc):
    """
    Class for performing quasi-harmonic approximation calculations.

    This class utilizes phonopy and Pymatgen-based structure manipulation to calculate
    thermal properties such as Gibbs free energy, thermal expansion, heat capacity, and
    bulk modulus as a function of temperature under the quasi-harmonic approximation.
    It allows for structural relaxation, handling customized scale factors for lattice constants,
    and fine-tuning various calculation parameters. Calculation results can be selectively
    saved to output files.

    :ivar calculator: Calculator instance used for electronic structure or energy calculations.
    :type calculator: Calculator
    :ivar t_step: Temperature step size in Kelvin.
    :type t_step: float
    :ivar t_max: Maximum temperature in Kelvin.
    :type t_max: float
    :ivar t_min: Minimum temperature in Kelvin.
    :type t_min: float
    :type pressure: float
    :ivar fmax: Maximum force threshold for structure relaxation in eV/Å.
    :type fmax: float
    :ivar optimizer: Type of optimizer used for structural relaxation.
    :type optimizer: str
    :ivar eos: Equation of state used for fitting energy vs. volume data.
    :type eos: Literal["vinet", "birch_murnaghan", "murnaghan"]
    :ivar relax_structure: Whether to perform structure relaxation before phonon calculations.
    :type relax_structure: bool
    :ivar relax_calc_kwargs: Additional keyword arguments for structure relaxation calculations.
    :type relax_calc_kwargs: dict | None
    :ivar phonon_calc_kwargs: Additional keyword arguments for phonon calculations.
    :type phonon_calc_kwargs: dict | None
    :ivar scale_factors: List of scale factors for lattice scaling.
    :type scale_factors: Sequence[float]
    :ivar write_helmholtz_volume: Path or boolean to control saving Helmholtz free energy vs. volume data.
    :type write_helmholtz_volume: bool | str | Path
    :ivar write_volume_temperature: Path or boolean to control saving volume vs. temperature data.
    :type write_volume_temperature: bool | str | Path
    :ivar write_thermal_expansion: Path or boolean to control saving thermal expansion coefficient data.
    :type write_thermal_expansion: bool | str | Path
    :ivar write_gibbs_temperature: Path or boolean to control saving Gibbs free energy as a function of temperature.
    :type write_gibbs_temperature: bool | str | Path
    :ivar write_bulk_modulus_temperature: Path or boolean to control saving bulk modulus vs. temperature data.
    :type write_bulk_modulus_temperature: bool | str | Path
    :ivar write_heat_capacity_p_numerical: Path or boolean to control saving numerically calculated heat capacity vs.
        temperature data.
    :type write_heat_capacity_p_numerical: bool | str | Path
    :ivar write_heat_capacity_p_polyfit: Path or boolean to control saving polynomial-fitted heat capacity vs.
        temperature data.
    :type write_heat_capacity_p_polyfit: bool | str | Path
    :ivar write_gruneisen_temperature: Path or boolean to control saving Grüneisen parameter vs. temperature data.
    :type write_gruneisen_temperature: bool | str | Path
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 1e-5,
        pressure: None | float = None,
        optimizer: str = "FIRE",
        eos: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        phonon_calc_kwargs: dict | None = None,
        scale_factors: Sequence[float] = tuple(np.arange(0.95, 1.05, 0.01)),
        write_helmholtz_volume: bool | str | Path = False,
        write_volume_temperature: bool | str | Path = False,
        write_thermal_expansion: bool | str | Path = False,
        write_gibbs_temperature: bool | str | Path = False,
        write_bulk_modulus_temperature: bool | str | Path = False,
        write_heat_capacity_p_numerical: bool | str | Path = False,
        write_heat_capacity_p_polyfit: bool | str | Path = False,
        write_gruneisen_temperature: bool | str | Path = False,
    ) -> None:
        """
        Initializes the class that handles thermal and structural calculations, including atomic
        structure relaxation, property evaluations, and phononic calculations across temperature
        ranges. This class is mainly designed to facilitate systematic computations involving
        temperature-dependent material properties and thermodynamic quantities.

        :param calculator: Calculator object or string indicating the computational engine to use
            for performing calculations.
        :param t_step: Step size for the temperature range, given in units of K.
        :param t_max: Maximum temperature for the calculations, given in units of K.
        :param t_min: Minimum temperature for the calculations, given in units of K.
        :param pressure: Pressure to calculate thermochemistry at, given in units of GPa.
        :param fmax: Maximum force convergence criterion for structure relaxation, in force units.
        :param optimizer: Name of the optimizer to use for structure optimization, default is
            "FIRE".
        :param eos: Equation of state to use for calculating energy vs. volume relationships.
            Default is "vinet".
        :param relax_structure: A boolean flag indicating whether the atomic structure should be
            relaxed as part of the computation workflow.
        :param relax_calc_kwargs: A dictionary containing additional keyword arguments to pass to
            the relax calculation.
        :param phonon_calc_kwargs: A dictionary containing additional parameters to pass to the
            phonon calculation routine.
        :param scale_factors: A sequence of scale factors for volume scaling during
            thermodynamic and phononic calculations.
        :param write_helmholtz_volume: Path, boolean, or string to indicate whether and where
            to save Helmholtz energy as a function of volume.
        :param write_volume_temperature: Path, boolean, or string to indicate whether and where
            to save equilibrium volume as a function of temperature.
        :param write_thermal_expansion: Path, boolean, or string to indicate whether and where
            to save the thermal expansion coefficient as a function of temperature.
        :param write_gibbs_temperature: Path, boolean, or string to indicate whether and where
            to save Gibbs energy as a function of temperature.
        :param write_bulk_modulus_temperature: Path, boolean, or string to indicate whether and
            where to save bulk modulus as a function of temperature.
        :param write_heat_capacity_p_numerical: Path, boolean, or string to indicate whether and
            where to save specific heat capacity at constant pressure, calculated numerically.
        :param write_heat_capacity_p_polyfit: Path, boolean, or string to indicate whether and
            where to save specific heat capacity at constant pressure, calculated via polynomial
            fitting.
        :param write_gruneisen_temperature: Path, boolean, or string to indicate whether and
            where to save Grüneisen parameter values as a function of temperature.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.pressure = pressure
        self.fmax = fmax
        self.optimizer = optimizer
        self.eos = eos
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.phonon_calc_kwargs = phonon_calc_kwargs
        self.scale_factors = scale_factors
        self.write_helmholtz_volume = write_helmholtz_volume
        self.write_volume_temperature = write_volume_temperature
        self.write_thermal_expansion = write_thermal_expansion
        self.write_gibbs_temperature = write_gibbs_temperature
        self.write_bulk_modulus_temperature = write_bulk_modulus_temperature
        self.write_heat_capacity_p_numerical = write_heat_capacity_p_numerical
        self.write_heat_capacity_p_polyfit = write_heat_capacity_p_polyfit
        self.write_gruneisen_temperature = write_gruneisen_temperature
        for key, val, default_path in (
            (
                "write_helmholtz_volume",
                self.write_helmholtz_volume,
                "helmholtz_volume.dat",
            ),
            (
                "write_volume_temperature",
                self.write_volume_temperature,
                "volume_temperature.dat",
            ),
            (
                "write_thermal_expansion",
                self.write_thermal_expansion,
                "thermal_expansion.dat",
            ),
            (
                "write_gibbs_temperature",
                self.write_gibbs_temperature,
                "gibbs_temperature.dat",
            ),
            (
                "write_bulk_modulus_temperature",
                self.write_bulk_modulus_temperature,
                "bulk_modulus_temperature.dat",
            ),
            (
                "write_heat_capacity_p_numerical",
                self.write_heat_capacity_p_numerical,
                "Cp_temperature.dat",
            ),
            (
                "write_heat_capacity_p_polyfit",
                self.write_heat_capacity_p_polyfit,
                "Cp_temperature_polyfit.dat",
            ),
            (
                "write_gruneisen_temperature",
                self.write_gruneisen_temperature,
                "gruneisen_temperature.dat",
            ),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Calculates thermal properties of Pymatgen structure with phonopy under quasi-harmonic approximation.

        Args:
            structure: Pymatgen structure.

        Returns:
        {
            "qha": Phonopy.qha object,
            "scale_factors": List of scale factors of lattice constants,
            "volumes": List of unit cell volumes at corresponding scale factors (in Angstrom^3),
            "electronic_energies": List of electronic energies at corresponding volumes (in eV),
            "temperatures": List of temperatures in ascending order (in Kelvin),
            "thermal_expansion_coefficients": List of volumetric thermal expansion coefficients at corresponding
                temperatures (in Kelvin^-1),
            "gibbs_free_energies": List of Gibbs free energies at corresponding temperatures (in eV),
            "bulk_modulus_P": List of bulk modulus at constant pressure at corresponding temperatures (in GPa),
            "heat_capacity_P": List of heat capacities at constant pressure at corresponding temperatures (in J/K/mol),
            "gruneisen_parameters": List of Gruneisen parameters at corresponding temperatures,
        }
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relax_calc_kwargs = {"fmax": self.fmax, "optimizer": self.optimizer, "max_steps": 5000} | (
                self.relax_calc_kwargs or {}
            )
            relaxer = RelaxCalc(self.calculator, **relax_calc_kwargs)
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        volumes, electronic_energies, free_energies, entropies, heat_capacities = self._collect_properties(structure_in)

        qha = self._create_qha(
            volumes,
            electronic_energies,
            temperatures,  # type: ignore[arg-type]
            free_energies,
            entropies,
            heat_capacities,
            self.pressure,
        )

        self._write_output_files(qha)

        return result | self._generate_output_dict(qha, volumes, electronic_energies, temperatures)  # type: ignore[arg-type]

    def _collect_properties(self, structure: Structure) -> tuple[list, list, list, list, list]:
        """Helper to collect properties like volumes, electronic energies, and thermal properties.

        Args:
            structure: Pymatgen structure for which the properties need to be calculated.

        Returns:
            Tuple containing lists of volumes, electronic energies, free energies, entropies,
                and heat capacities for different scale factors.
        """
        volumes = []
        electronic_energies = []
        free_energies = []
        entropies = []
        heat_capacities = []
        for scale_factor in self.scale_factors:
            struct = self._scale_structure(structure, scale_factor)
            volumes.append(struct.volume)
            electronic_energies.append(run_pes_calc(struct, self.calculator).energy)
            thermal_properties = self._calculate_thermal_properties(struct)
            free_energies.append(thermal_properties["free_energy"])
            entropies.append(thermal_properties["entropy"])
            heat_capacities.append(thermal_properties["heat_capacity"])
        return volumes, electronic_energies, free_energies, entropies, heat_capacities

    def _scale_structure(self, structure: Structure, scale_factor: float) -> Structure:
        """Helper to scale the lattice of a structure.

        Args:
            structure: Pymatgen structure to be scaled.
            scale_factor: Factor by which the lattice constants are scaled.

        Returns:
            Pymatgen structure with scaled lattice constants.
        """
        struct = structure.copy()
        struct.apply_strain(scale_factor - 1)
        return struct

    def _calculate_thermal_properties(self, structure: Structure) -> dict:
        """Helper to calculate the thermal properties of a structure.

        Args:
            structure: Pymatgen structure for which the thermal properties are calculated.

        Returns:
            Dictionary of thermal properties containing free energies, entropies and heat capacities.
        """
        phonon_calc_kwargs = {
            "t_step": self.t_step,
            "t_max": self.t_max,
            "t_min": self.t_min,
            "relax_structure": True,
            "relax_calc_kwargs": {"relax_cell": False},
            "write_phonon": False,
        } | (self.phonon_calc_kwargs or {})
        phonon_calc = PhononCalc(
            self.calculator,
            **phonon_calc_kwargs,
        )
        return phonon_calc.calc(structure)["thermal_properties"]

    def _create_qha(
        self,
        volumes: list,
        electronic_energies: list,
        temperatures: list,
        free_energies: list,
        entropies: list,
        heat_capacities: list,
        pressure: None | float,
    ) -> PhonopyQHA:
        """Helper to create a PhonopyQHA object for quasi-harmonic approximation.

        Args:
            volumes: List of volumes corresponding to different scale factors.
            electronic_energies: List of electronic energies corresponding to different volumes.
            temperatures: List of temperatures in ascending order (in Kelvin).
            free_energies: List of free energies corresponding to different volumes and temperatures.
            entropies: List of entropies corresponding to different volumes and temperatures.
            heat_capacities: List of heat capacities corresponding to different volumes and temperatures.
            pressure: Pressure in GPa.

        Returns:
            Phonopy.qha object.
        """
        return PhonopyQHA(
            volumes=volumes,
            electronic_energies=electronic_energies,
            temperatures=temperatures,
            free_energy=np.transpose(free_energies),
            cv=np.transpose(heat_capacities),
            entropy=np.transpose(entropies),
            pressure=pressure,
            eos=self.eos,
            t_max=self.t_max,
        )

    def _write_output_files(self, qha: PhonopyQHA) -> None:
        """Helper to write various output files based on the QHA calculation.

        Args:
            qha: Phonopy.qha object
        """
        if self.write_helmholtz_volume:
            qha.write_helmholtz_volume(filename=self.write_helmholtz_volume)
        if self.write_volume_temperature:
            qha.write_volume_temperature(filename=self.write_volume_temperature)
        if self.write_thermal_expansion:
            qha.write_thermal_expansion(filename=self.write_thermal_expansion)
        if self.write_gibbs_temperature:
            qha.write_gibbs_temperature(filename=self.write_gibbs_temperature)
        if self.write_bulk_modulus_temperature:
            qha.write_bulk_modulus_temperature(filename=self.write_bulk_modulus_temperature)
        if self.write_heat_capacity_p_numerical:
            qha.write_heat_capacity_P_numerical(filename=self.write_heat_capacity_p_numerical)
        if self.write_heat_capacity_p_polyfit:
            qha.write_heat_capacity_P_polyfit(filename=self.write_heat_capacity_p_polyfit)
        if self.write_gruneisen_temperature:
            qha.write_gruneisen_temperature(filename=self.write_gruneisen_temperature)

    def _generate_output_dict(
        self, qha: PhonopyQHA, volumes: list, electronic_energies: list, temperatures: list
    ) -> dict:
        """Helper to generate the output dictionary after QHA calculation.

        Args:
            qha: Phonopy.qha object.
            volumes: List of volumes corresponding to different scale factors.
            electronic_energies: List of electronic energies corresponding to different volumes.
            temperatures: List of temperatures in ascending order (in Kelvin).

        Returns:
            Dictionary containing the results of QHA calculation.
        """
        return {
            "qha": qha,
            "scale_factors": self.scale_factors,
            "volumes": volumes,
            "electronic_energies": electronic_energies,
            "temperatures": temperatures,
            "thermal_expansion_coefficients": qha.thermal_expansion,
            "gibbs_free_energies": qha.gibbs_temperature,
            "bulk_modulus_P": qha.bulk_modulus_temperature,
            "heat_capacity_P": qha.heat_capacity_P_polyfit,
            "gruneisen_parameters": qha.gruneisen_temperature,
        }
