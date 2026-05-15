"""Calculator for phonon properties under quasi-harmonic approximation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from phonopy import PhonopyQHA

from ._base import PropCalc
from ._phonon import PhononCalc
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence
    from typing import Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class QHACalc(PropCalc):
    """
    Class for performing quasi-harmonic approximation calculations.

    This class utilizes phonopy and Pymatgen-based structure manipulation to calculate
    thermal properties such as Gibbs free energy, thermal expansion, heat capacity, and
    bulk modulus as a function of temperature under the quasi-harmonic approximation.
    It allows for structural relaxation, handling customized scale factors for lattice constants,
    and fine-tuning various calculation parameters. Calculation results can be selectively
    saved to output files.

    Attributes:
        calculator: ASE calculator or universal model name.
        t_step, t_max, t_min: Temperature grid (K) for QHA output.
        pressure: Target pressure(s) in GPa. Accepts a single float, a list of floats, or
            None. When multiple pressures are given the expensive phonon calculations are run
            once and the cheap QHA fitting is repeated per pressure.
        fmax: Relaxation force tolerance (eV/Å).
        max_steps: Max relaxation steps.
        optimizer: ASE optimizer name.
        eos: EOS model for volume-energy fits (``vinet``, ``birch_murnaghan``, ``murnaghan``).
        allow_shape_change: Allow cell shape change at fixed volume in scaled relaxations.
        relax_structure: Relax initial structure before the QHA volume scan.
        relax_calc_kwargs: Kwargs merged into all ``RelaxCalc`` uses.
        phonon_calc_kwargs: Kwargs merged into ``PhononCalc``.
        scale_factors: Volume scale factors for the QHA mesh.
        imaginary_freq_tol, on_imaginary_modes, fix_imaginary_attempts: Passed to ``PhononCalc``.
        store_ha_phonon: Whether to store per-scale-factor harmonic approximation results in
            the output dict under the ``"ha"`` key.
        write_ha_phonon: Write per-scale-factor phonopy files. False disables writing,
            True uses the default name ``phonon_{scale_factor:.3f}.yaml``, or a format string
            with a ``{scale_factor}`` placeholder selects a custom name.
        write_*: Output paths (or True for defaults) for phonopy QHA text files.
        write_*: Output paths (or True for defaults) for phonopy QHA text files. When multiple
            pressures are requested a ``_P{pressure}GPa`` suffix is inserted before the file
            extension to avoid overwriting files across pressures.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        pressure: None | float | Sequence[float] = None,
        fmax: float = 1e-5,
        max_steps: int = 5000,
        optimizer: str = "FIRE",
        eos: Literal["vinet", "birch_murnaghan", "murnaghan"] = "vinet",
        allow_shape_change: bool = True,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        phonon_calc_kwargs: dict | None = None,
        scale_factors: Sequence[float] = (0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05),
        imaginary_freq_tol: float = -0.01,
        on_imaginary_modes: Literal["error", "warn"] = "warn",
        fix_imaginary_attempts: int = 0,
        store_ha_phonon: bool = True,
        write_ha_phonon: bool | str = False,
        write_helmholtz_volume: bool | str | os.PathLike = False,
        write_volume_temperature: bool | str | os.PathLike = False,
        write_thermal_expansion: bool | str | os.PathLike = False,
        write_gibbs_temperature: bool | str | os.PathLike = False,
        write_bulk_modulus_temperature: bool | str | os.PathLike = False,
        write_heat_capacity_p_numerical: bool | str | os.PathLike = False,
        write_heat_capacity_p_polyfit: bool | str | os.PathLike = False,
        write_gruneisen_temperature: bool | str | os.PathLike = False,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            t_step: Temperature sampling step (K).
            t_max: Maximum temperature (K).
            t_min: Minimum temperature (K).
            pressure: Pressure in GPa for Gibbs terms. Accepts a single float, a list of
                floats, or None. When multiple pressures are supplied the phonon calculations
                are performed only once and the QHA fit is repeated for each pressure.
            fmax: Force tolerance for relaxations.
            max_steps: Max steps per relaxation.
            optimizer: ASE optimizer name.
            eos: EOS key for ``PhonopyQHA``.
            allow_shape_change: Allow shape relaxation at constant volume for scaled cells.
            relax_structure: Relax input once before scanning ``scale_factors``.
            relax_calc_kwargs: Extra kwargs for every ``RelaxCalc``.
            phonon_calc_kwargs: Extra kwargs for each ``PhononCalc``.
            scale_factors: Volume scale factors applied to the relaxed structure.
            imaginary_freq_tol: Imaginary-mode threshold (THz) for ``PhononCalc``.
            on_imaginary_modes: ``"warn"`` or ``"error"``.
            fix_imaginary_attempts: Imaginary-mode retries per scale factor in ``PhononCalc``.
            store_ha_phonon: If True (default), each per-scale-factor ``PhononCalc`` result
                dict is stored and returned under the ``"ha"`` key. Set to False to skip
                storing these results and omit the ``"ha"`` key from the output entirely.
            write_ha_phonon: Write a phonopy file for each scale factor. False (default)
                disables writing. True writes ``phonon_{scale_factor:.3f}.yaml``. A format
                string containing ``{scale_factor}`` selects a custom filename pattern.
            write_helmholtz_volume: Write F(V); True selects default filename.
            write_volume_temperature: Write V(T).
            write_thermal_expansion: Write thermal expansion alpha(T).
            write_gibbs_temperature: Write G(T).
            write_bulk_modulus_temperature: Write B(T).
            write_heat_capacity_p_numerical: Write Cp(T) numerical.
            write_heat_capacity_p_polyfit: Write Cp(T) polyfit.
            write_gruneisen_temperature: Write Gruneisen gamma(T).
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.eos = eos
        self.allow_shape_change = allow_shape_change
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.phonon_calc_kwargs = phonon_calc_kwargs
        # EOS fitting downstream assumes monotonic volume ordering.
        self.scale_factors = tuple(sorted(scale_factors))
        self.imaginary_freq_tol = imaginary_freq_tol
        self.on_imaginary_modes = on_imaginary_modes
        self.fix_imaginary_attempts = fix_imaginary_attempts
        self.store_ha_phonon = store_ha_phonon
        self.write_ha_phonon = write_ha_phonon

        # Normalize pressure to an internal list; None becomes [None].
        self.pressure = pressure
        if pressure is None:
            self.pressures: list[float | None] = [None]
        elif isinstance(pressure, int | float):
            self.pressures = [float(pressure)]
        else:
            self.pressures = list(pressure)

        # Needed to make sure the volume doesn't change during relaxations on scaled structures
        self._fixed_cell_relax_calc_kwargs = {
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            **(self.relax_calc_kwargs or {}),
        }
        self._fixed_cell_relax_calc_kwargs["relax_cell"] = bool(self.allow_shape_change)
        self._fixed_cell_relax_calc_kwargs["cell_filter_kwargs"] = (
            {"constant_volume": True} if self.allow_shape_change else {}
        )

        # Normalize write_* inputs to Optional[str | os.PathLike]:
        # - True  -> default filename (meaning "write to default file")
        # - False -> None (disabled)
        # - str/PathLike -> keep as-is (user-provided path)
        self.write_helmholtz_volume: str | os.PathLike | None = None
        self.write_volume_temperature: str | os.PathLike | None = None
        self.write_thermal_expansion: str | os.PathLike | None = None
        self.write_gibbs_temperature: str | os.PathLike | None = None
        self.write_bulk_modulus_temperature: str | os.PathLike | None = None
        self.write_heat_capacity_p_numerical: str | os.PathLike | None = None
        self.write_heat_capacity_p_polyfit: str | os.PathLike | None = None
        self.write_gruneisen_temperature: str | os.PathLike | None = None

        for key, val, default_path in (
            ("write_helmholtz_volume", write_helmholtz_volume, "helmholtz_volume.dat"),
            ("write_volume_temperature", write_volume_temperature, "volume_temperature.dat"),
            ("write_thermal_expansion", write_thermal_expansion, "thermal_expansion.dat"),
            ("write_gibbs_temperature", write_gibbs_temperature, "gibbs_temperature.dat"),
            ("write_bulk_modulus_temperature", write_bulk_modulus_temperature, "bulk_modulus_temperature.dat"),
            ("write_heat_capacity_p_numerical", write_heat_capacity_p_numerical, "Cp_temperature.dat"),
            ("write_heat_capacity_p_polyfit", write_heat_capacity_p_polyfit, "Cp_temperature_polyfit.dat"),
            ("write_gruneisen_temperature", write_gruneisen_temperature, "gruneisen_temperature.dat"),
        ):
            if val is True:
                normalized: str | os.PathLike | None = default_path
            elif val is False:
                normalized = None
            else:
                normalized = val
            setattr(self, key, normalized)

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Quasi-harmonic thermodynamics via phonopy over a volume scan.

        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with per-volume data (``scale_factors``, ``volumes`` in A^3,
            ``electronic_energies`` in eV, ``scaled_structures``), a ``pressures`` list (GPa),
            ``temperatures`` (K), and a ``qha_results`` list of per-pressure dicts each
            containing ``pressure`` (GPa), ``qha`` (``PhonopyQHA``),
            ``thermal_expansion_coefficients`` (1/K), ``gibbs_free_energies`` (kJ/mol),
            ``bulk_modulus_P`` (GPa), ``heat_capacity_P`` (J/(K*mol)), and
            ``gruneisen_parameters`` (dimensionless). When only a single pressure was
            requested the top-level dict also contains those keys directly for backward
            compatibility. When ``store_ha_phonon=True``, also includes ``"ha"``: a list of
            per-scale-factor ``PhononCalc`` result dicts. ``_units`` maps each numeric output
            to its unit string. Merged with any relaxation fields from earlier steps.
        """
        result = super().calc(structure)
        if self.relax_structure:
            logger.info("Relaxing input structure before QHA")
        result, raw_structure = self._prerelax(
            result["final_structure"],
            result,
            fmax=self.fmax,
            optimizer=self.optimizer,
            max_steps=self.max_steps,
        )
        structure_in: Structure = to_pmg_structure(raw_structure)

        logger.info(
            "Starting QHA calculation over %d scale factors: %s",
            len(self.scale_factors),
            list(self.scale_factors),
        )
        properties = self._collect_properties(structure_in)

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        # PhonopyQHA needs one extra temperature point for finite-difference derivatives.
        temperatures_ext = np.append(temperatures, temperatures[-1] + self.t_step)

        qha_results: list[dict] = []
        for pressure in self.pressures:
            logger.info(
                "Fitting equation of state and computing QHA thermal properties at pressure=%s GPa",
                pressure,
            )
            qha = PhonopyQHA(
                volumes=properties["volumes"],
                electronic_energies=properties["electronic_energies"],
                temperatures=temperatures_ext,
                free_energy=np.transpose(properties["free_energies"]),
                cv=np.transpose(properties["heat_capacities"]),
                entropy=np.transpose(properties["entropies"]),
                pressure=pressure,
                eos=self.eos,
            )
            self._write_output_files(qha, pressure=pressure)
            qha_results.append(
                {
                    "pressure": pressure,
                    "qha": qha,
                    "thermal_expansion_coefficients": qha.thermal_expansion,
                    "gibbs_free_energies": qha.gibbs_temperature,
                    "bulk_modulus_P": qha.bulk_modulus_temperature,
                    "heat_capacity_P": qha.heat_capacity_P_polyfit,
                    "gruneisen_parameters": qha.gruneisen_temperature,
                }
            )

        output_dict: dict[str, Any] = {
            "pressures": self.pressures,
            "temperatures": temperatures,
            "qha_results": qha_results,
            "scale_factors": self.scale_factors,
            "volumes": properties["volumes"],
            "scaled_structures": properties["scaled_structures"],
            "electronic_energies": properties["electronic_energies"],
        }

        if self.store_ha_phonon:
            output_dict["ha"] = properties["ha"]
        # Backward-compatible unwrap for the single-pressure case.
        if len(self.pressures) == 1:
            output_dict |= qha_results[0]

        output_dict["_units"] = {
            **result.get("_units", {}),
            "pressures": "GPa",
            "temperatures": "K",
            "volumes": "A^3",
            "electronic_energies": "eV",
            "thermal_expansion_coefficients": "1/K",
            "gibbs_free_energies": "kJ/mol",
            "bulk_modulus_P": "GPa",
            "heat_capacity_P": "J/(K*mol)",
            "gruneisen_parameters": "dimensionless",
        }

        return result | output_dict

    def _collect_properties(self, structure: Structure) -> dict[str, list]:
        """Helper to collect properties like volumes, electronic energies, and thermal properties.

        Args:
            structure: Primitive cell structure at the reference volume.

        Returns:
            Dict of lists keyed by ``volumes``, ``electronic_energies``, ``free_energies``,
            ``entropies``, ``heat_capacities``, ``scaled_structures``, and ``ha``
            (per-scale-factor ``PhononCalc`` result dicts).
        """
        volumes = []
        electronic_energies = []
        free_energies = []
        entropies = []
        heat_capacities = []
        scaled_structures = []
        ha = []
        for scale_factor in self.scale_factors:
            # Apply linear strain
            scaled_structure = self._scale_structure(structure, scale_factor)

            # Relax at fixed volume
            logger.info("Scale factor %.3f: relaxing at fixed volume", scale_factor)
            relaxer = RelaxCalc(self.calculator, **self._fixed_cell_relax_calc_kwargs)
            relaxed_result = relaxer.calc(scaled_structure)

            # Resolve the phonon write path for this scale factor
            if self.write_ha_phonon is False:
                write_phonon: bool | str = False
            elif self.write_ha_phonon is True:
                write_phonon = f"phonon_{scale_factor:.3f}.yaml"
            else:
                write_phonon = str(self.write_ha_phonon).format(scale_factor=scale_factor)

            # Calculate thermal properties from phonon calculation and tabulate results.
            # We must store the energy, structure, etc. from the phonon calculation if
            # available in case there are any correction attempts made to resolve imaginary modes,
            # which would update the relaxation output.
            logger.info(
                "Scale factor %.3f: computing phonon thermal properties (V=%.1f Å³)",
                scale_factor,
                relaxed_result["final_structure"].volume,
            )
            phonon_result = self._calculate_thermal_properties(
                relaxed_result["final_structure"], write_phonon=write_phonon
            )

            # Collect properties
            volume = phonon_result["final_structure"].volume
            volume_drift = np.abs(volume - scaled_structure.volume) / scaled_structure.volume
            if volume_drift > 1e-4:  # noqa: PLR2004
                logger.warning(
                    "Scale factor %.3f: fixed-cell relaxation drifted volume by %.4f%% "
                    "(expected %.4f Å³, got %.4f Å³). Skipping this scale factor; the QHA "
                    "fit will use the remaining points. Loosen fmax or tighten the cell "
                    "filter constraints if this happens often.",
                    scale_factor,
                    volume_drift * 100,
                    scaled_structure.volume,
                    volume,
                )
                continue
            ha.append(phonon_result)
            scaled_structures.append(phonon_result["final_structure"])
            volumes.append(volume)
            electronic_energies.append(phonon_result.get("energy", relaxed_result.get("energy")))
            thermal_properties = phonon_result["thermal_properties"]
            free_energies.append(thermal_properties["free_energy"])
            entropies.append(thermal_properties["entropy"])
            heat_capacities.append(thermal_properties["heat_capacity"])
        return {
            "volumes": volumes,
            "electronic_energies": electronic_energies,
            "free_energies": free_energies,
            "entropies": entropies,
            "heat_capacities": heat_capacities,
            "scaled_structures": scaled_structures,
            "ha": ha,
        }

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

    def _calculate_thermal_properties(self, structure: Structure, *, write_phonon: bool | str = False) -> dict:
        """Helper to calculate the thermal properties of a structure.

        Args:
            structure: Pymatgen structure for which the thermal properties are calculated.
            write_phonon: Passed directly to ``PhononCalc`` as ``write_phonon``. False disables
                writing, True uses the phonopy default name, or a string path is used as-is.

        Returns:
            Full result dict from PhononCalc, containing "energy" (eV, from the
            volume-fixed ionic relaxation) and "thermal_properties" (free energies,
            entropies and heat capacities).
        """
        phonon_calc_kwargs = {
            "t_step": self.t_step,
            "t_max": self.t_max + self.t_step,  # one extra point for PhonopyQHA finite differences
            "t_min": self.t_min,
            "relax_calc_kwargs": self._fixed_cell_relax_calc_kwargs,  # needed for _resolve_imaginary_modes
            "imaginary_freq_tol": self.imaginary_freq_tol,
            "on_imaginary_modes": self.on_imaginary_modes,
            "fix_imaginary_attempts": self.fix_imaginary_attempts,
            "write_phonon": write_phonon,
        } | (self.phonon_calc_kwargs or {})
        phonon_calc_kwargs["relax_structure"] = False  # already relaxed
        phonon_calc = PhononCalc(
            self.calculator,
            **cast("Any", phonon_calc_kwargs),
        )
        return phonon_calc.calc(structure)

    def _write_output_files(self, qha: PhonopyQHA, pressure: float | None = None) -> None:  # noqa: C901
        """Helper to write various output files based on the QHA calculation.

        When multiple pressures are requested, a ``_P{pressure}GPa`` suffix is inserted
        before the file extension to avoid overwriting files across pressures.

        Args:
            qha: Phonopy.qha object.
            pressure: The pressure (GPa) associated with this QHA fit, or None.
        """

        def _suffixed(path: str | os.PathLike) -> str | os.PathLike:
            """Insert a pressure suffix before the file extension, only for multi-pressure runs."""
            if pressure is None or len(self.pressures) == 1:
                return path
            s = str(path)
            stem, _, ext = s.rpartition(".")
            return f"{stem}_P{pressure}GPa.{ext}" if stem else f"{s}_P{pressure}GPa"

        # write_* are Optional[str | os.PathLike]; None means "do not write".
        if self.write_helmholtz_volume is not None:
            qha.write_helmholtz_volume(filename=_suffixed(self.write_helmholtz_volume))
        if self.write_volume_temperature is not None:
            qha.write_volume_temperature(filename=_suffixed(self.write_volume_temperature))
        if self.write_thermal_expansion is not None:
            qha.write_thermal_expansion(filename=_suffixed(self.write_thermal_expansion))
        if self.write_gibbs_temperature is not None:
            qha.write_gibbs_temperature(filename=_suffixed(self.write_gibbs_temperature))
        if self.write_bulk_modulus_temperature is not None:
            qha.write_bulk_modulus_temperature(filename=_suffixed(self.write_bulk_modulus_temperature))
        if self.write_heat_capacity_p_numerical is not None:
            qha.write_heat_capacity_P_numerical(filename=_suffixed(self.write_heat_capacity_p_numerical))
        if self.write_heat_capacity_p_polyfit is not None:
            qha.write_heat_capacity_P_polyfit(filename=_suffixed(self.write_heat_capacity_p_polyfit))
        if self.write_gruneisen_temperature is not None:
            qha.write_gruneisen_temperature(filename=_suffixed(self.write_gruneisen_temperature))
