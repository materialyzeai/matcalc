"""Calculator for phonon properties."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_ase_atoms, to_pmg_structure

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class PhononCalc(PropCalc):
    """
    A class for phonon and thermal property calculations using phonopy.

    The `PhononCalc` class extends the `PropCalc` class to provide
    functionalities for calculating phonon properties and thermal properties
    of a given structure using the phonopy library. It includes options for
    structural relaxation before phonon property determination, as well as
    methods to export calculated properties to various output files for
    further analysis or visualization.

    Attributes:
        calculator: ASE calculator or universal model name.
        atom_disp: Finite-difference displacement for force constants (Å).
        min_length: Minimum supercell dimension (Å); used if ``supercell_matrix`` is None.
        supercell_matrix: Explicit supercell; overrides ``min_length`` when set.
        t_step, t_max, t_min: Thermal property temperature grid (K).
        fmax: Relaxation force tolerance when ``relax_structure`` is used.
        max_steps: Max relaxation steps.
        optimizer: ASE optimizer name for pre-phonon relaxation.
        relax_structure: Relax structure before phonopy.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        imaginary_freq_tol: Frequencies below this (THz) count as imaginary.
        on_imaginary_modes: ``"warn"`` or ``"error"`` on imaginary modes.
        fix_imaginary_attempts: Rattle/retry cycles to fix imaginary modes.
        symprec: symmetry precision used for spglib symmetry finding.
        write_force_constants: Output path for force constants (or False).
        write_band_structure: Band structure YAML path (or False).
        write_total_dos: Total DOS path (or False).
        write_phonon: Phonopy save path (e.g. ``phonon.yaml``) or False.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        min_length: float | None = 20.0,
        supercell_matrix: ArrayLike | None = None,
        t_step: float = 10,
        t_max: float = 1000,
        t_min: float = 0,
        fmax: float = 1e-5,
        max_steps: int = 5000,
        optimizer: str = "FIRE",
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        imaginary_freq_tol: float = -0.01,
        on_imaginary_modes: Literal["error", "warn"] = "warn",
        fix_imaginary_attempts: int = 0,
        symprec: float = 1e-5,
        write_force_constants: bool | str | Path = False,
        write_band_structure: bool | str | Path = False,
        write_total_dos: bool | str | Path = False,
        write_phonon: bool | str | Path = True,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            atom_disp: Displacement for phonopy finite differences.
            min_length: Minimum supercell size along each lattice direction (Å).
            supercell_matrix: Integer supercell matrix; if None, derived from ``min_length``.
            t_step: Temperature step for thermal properties (K).
            t_max: Maximum temperature (K).
            t_min: Minimum temperature (K).
            fmax: Relaxation force tolerance (eV/Å).
            max_steps: Maximum relaxation steps before phonopy.
            optimizer: Optimizer for pre-phonon relaxation.
            relax_structure: Whether to relax before building phonopy.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
            imaginary_freq_tol: Threshold (THz) for classifying imaginary modes.
            on_imaginary_modes: ``"warn"`` or ``"error"`` when imaginary modes exist.
            fix_imaginary_attempts: Rattle/relax/phonon retries; 0 disables.
            symprec: symmetry precision used for spglib symmetry finding.
            write_force_constants: Path to write FCs, True for default name, or False.
            write_band_structure: Path, True for default YAML, or False.
            write_total_dos: Path, True for default DOS file, or False.
            write_phonon: Path, True for default ``phonon.yaml``, or False.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.min_length = min_length
        self.supercell_matrix = supercell_matrix
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.imaginary_freq_tol = imaginary_freq_tol
        self.on_imaginary_modes = on_imaginary_modes
        self.fix_imaginary_attempts = fix_imaginary_attempts
        self.symprec = symprec
        self.write_force_constants = write_force_constants
        self.write_band_structure = write_band_structure
        self.write_total_dos = write_total_dos
        self.write_phonon = write_phonon

        if supercell_matrix is None and min_length is None:
            raise ValueError("min_length must be set when supercell_matrix is None.")

        # Set default paths for output files.
        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """Compute phonons and thermal properties with phonopy.

        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with ``phonon`` (``phonopy.Phonopy``), ``thermal_properties`` (phonopy thermal
            dict: ``temperatures`` in K, ``free_energy`` in kJ/mol, ``entropy`` in J/(K*mol),
            ``heat_capacity`` in J/(K*mol)), ``frequencies`` in THz, ``disp_supercells``, plus
            any relaxation fields merged in. ``_units`` maps each numeric output to its unit
            string.
        """
        result = super().calc(structure)
        structure_in: Structure = to_pmg_structure(result["final_structure"])

        if self.relax_structure:
            logger.info("Relaxing input structure before phonon calculation")
            result |= self._relax_structure(structure_in)
            structure_in = result["final_structure"]

        phonon, frequencies, disp_supercells = self._run_phonopy(structure_in)

        if self.fix_imaginary_attempts > 0 and np.any(frequencies < self.imaginary_freq_tol):
            relax_result, phonon, frequencies, disp_supercells = self._resolve_imaginary_modes(
                structure_in, phonon, frequencies, disp_supercells
            )
            result |= relax_result
            structure_in = result["final_structure"]

        self._check_imaginary_modes(frequencies)

        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)  # type: ignore[arg-type]
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=cast("str", self.write_band_structure))
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)  # type: ignore[arg-type]
        units_map = {
            **result.get("_units", {}),
            "thermal_properties.temperatures": "K",
            "thermal_properties.free_energy": "kJ/mol",
            "thermal_properties.entropy": "J/(K*mol)",
            "thermal_properties.heat_capacity": "J/(K*mol)",
            "frequencies": "THz",
        }
        return result | {
            "phonon": phonon,
            "thermal_properties": phonon.get_thermal_properties_dict(),
            "frequencies": frequencies,
            "disp_supercells": disp_supercells,
            "_units": units_map,
        }

    def _run_phonopy(self, structure: Structure) -> tuple[phonopy.Phonopy, np.ndarray, list[Structure]]:
        """Build a Phonopy object, run force calculations, and return (phonon, frequencies, disp_supercells).

        Args:
            structure: Pymatgen structure.

        Returns:
            Tuple of (phonon, frequencies, disp_supercells).
        """
        cell = get_phonopy_structure(structure)
        if self.supercell_matrix is not None:
            supercell_matrix = np.array(self.supercell_matrix, dtype=int)
        else:
            supercell_matrix = np.diag(np.ceil(self.min_length / np.array(structure.lattice.abc)).astype(int))

        phonon = phonopy.Phonopy(
            cell,
            supercell_matrix=supercell_matrix,
            symprec=self.symprec,
            primitive_matrix="P",
        )
        phonon.generate_displacements(distance=self.atom_disp)
        disp_supercells = [
            get_pmg_structure(supercell)
            for supercell in phonon.supercells_with_displacements  # type:ignore[union-attr]
            if supercell is not None
        ]
        logger.info("Computing forces on %d displaced supercells", len(disp_supercells))
        phonon.forces = [run_pes_calc(supercell, self.calculator).forces for supercell in disp_supercells]
        phonon.produce_force_constants()
        phonon.run_mesh(with_eigenvectors=True)
        mesh_dict = cast("dict[str, Any]", phonon.get_mesh_dict())
        frequencies = mesh_dict["frequencies"]
        return phonon, frequencies, disp_supercells

    def _check_imaginary_modes(self, frequencies: np.ndarray) -> None:
        """Warn or raise if imaginary modes are present, based on on_imaginary_modes setting.

        Args:
            frequencies: Frequencies array from the mesh run.
        """
        # In phonopy, imaginary frequencies are represented as negative values.
        imag_freq_mask = frequencies < self.imaginary_freq_tol
        if np.any(imag_freq_mask):
            n_imag = np.sum(imag_freq_mask)
            n_freqs = frequencies.size
            min_mode = np.min(frequencies)
            pct_imag = 100 * n_imag / n_freqs
            msg = (
                f"{n_imag}/{n_freqs} ({pct_imag:.12}%) modes are imaginary (below {self.imaginary_freq_tol:.4f} THz). "
                f"Most negative: {min_mode:.2f} THz. This indicates a dynamically unstable structure. "
                f"Thermal properties may not be reliable."
            )
            if self.on_imaginary_modes.lower() == "warn":
                logger.warning(msg)
            else:
                raise ValueError(msg)

    def _resolve_imaginary_modes(
        self,
        structure_in: Structure,
        phonon: phonopy.Phonopy,
        frequencies: np.ndarray,
        disp_supercells: list,
    ) -> tuple[dict, phonopy.Phonopy, np.ndarray, list[Structure]]:
        """Attempt to resolve imaginary modes by rattling and re-relaxing.

        Args:
            structure_in: Current primitive-cell structure.
            phonon: Phonopy object from the most recent build.
            frequencies: Frequencies from the most recent mesh run.
            disp_supercells: Displaced supercells from the most recent build.

        Returns:
            Updated (RelaxCalc dictionary, phonon, frequencies, disp_supercells).
        """
        logger.warning(
            "Imaginary modes detected (percent imaginary = %.2f%%, min freq: %.2f THz). "
            "Attempting to resolve over %d attempt(s).",
            100 * np.sum(frequencies < self.imaginary_freq_tol) / frequencies.size,
            np.min(frequencies),
            self.fix_imaginary_attempts,
        )
        relax_result: dict = {}
        for attempt in range(self.fix_imaginary_attempts):
            logger.info("Imaginary mode correction attempt %d/%d", attempt + 1, self.fix_imaginary_attempts)
            structure_in = self._rattle_structure(structure_in)

            logger.info("Re-relaxing structure at fixed cell volume following rattle.")
            relax_result = self._relax_structure(structure_in)
            structure_in = relax_result["final_structure"]

            logger.info("Recomputing phonons after correction attempt %d", attempt + 1)
            phonon, frequencies, disp_supercells = self._run_phonopy(structure_in)

            if np.any(frequencies < self.imaginary_freq_tol):
                logger.info(
                    "Imaginary modes persist after attempt %d (percent imaginary = %.2f%%, min freq: %.2f THz).",
                    attempt + 1,
                    100 * np.sum(frequencies < self.imaginary_freq_tol) / frequencies.size,
                    np.min(frequencies),
                )
            else:
                logger.info("Imaginary modes resolved after %d attempt(s)", attempt + 1)
                break
        return relax_result, phonon, frequencies, disp_supercells

    def _rattle_structure(self, structure_in: Structure, stdev: float = 0.01) -> Structure:
        """Rattle the atoms to bump out of stationary point (stdev=0.01 Å).

        Args:
            structure_in: Pymatgen structure.
            stdev: standard deviation in angstrom for the rattle.

        Returns:
            Pymatgen structure with rattled atomic positions.
        """
        atoms = to_ase_atoms(structure_in)
        atoms.rattle(stdev=stdev)
        atoms.wrap()
        return to_pmg_structure(atoms)

    def _relax_structure(self, structure_in: Structure) -> dict:
        """Relax a structure.

        Args:
            structure_in: Pymatgen structure.

        Returns:
            Full result dict from RelaxCalc.
        """
        relax_calc_kwargs = {"fmax": self.fmax, "optimizer": self.optimizer, "max_steps": self.max_steps} | (
            self.relax_calc_kwargs or {}
        )
        relaxer = RelaxCalc(self.calculator, **cast("Any", relax_calc_kwargs))
        return relaxer.calc(structure_in)
