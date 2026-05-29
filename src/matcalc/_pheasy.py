"""Calculator for phonon properties using pheasy."""

from __future__ import annotations

import logging
import pickle
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.file_IO import write_FORCE_CONSTANTS as write_force_constants
from phonopy.interface.vasp import write_vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from ._base import PropCalc
from ._relaxation import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class PheasyCalc(PropCalc):
    """
    Phonon and higher-order force-constant calculations driven by pheasy.

    Extends ``PropCalc`` to compute harmonic (and optionally anharmonic) force
    constants via the pheasy CLI, with optional pre-relaxation. Harmonic FCs
    are written to ``FORCE_CONSTANTS_2ND`` for downstream use (e.g. ShengBTE,
    FourPhonon).

    Attributes:
        calculator: ASE calculator or universal model name.
        atom_disp: Harmonic finite-difference displacement (Å).
        atom_disp_anhar: Anharmonic finite-difference displacement (Å).
        supercell_matrix: 3x3 supercell matrix; derived from ``min_length``
            when None.
        t_step, t_max, t_min: Thermal property temperature grid (K).
        fmax: Relaxation force tolerance (eV/Å).
        optimizer: ASE optimizer name for pre-relaxation.
        relax_structure: Relax structure before phonon calculation.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        write_force_constants: Output path for force constants, or False.
        write_band_structure: Output path for band structure YAML, or False.
        write_total_dos: Output path for total DOS data, or False.
        write_phonon: Output path for phonopy save file, or False.
        fitting_method: One of ``"FDM"``, ``"LASSO"``, ``"MD"``.
        num_harmonic_snapshots: Snapshots for harmonic fitting; defaults to
            2x the number of generated displacements.
        num_anharmonic_snapshots: Snapshots for anharmonic fitting; defaults
            to 10x the number of generated displacements.
        calc_anharmonic: Whether to compute anharmonic force constants.
        symprec: Symmetry precision for spglib.
        min_length: Minimum supercell length (Å).
        force_diagonal: Force a diagonal supercell transformation.
        cutoff_distance_cubic: Cutoff for cubic force constants (Å).
        cutoff_distance_quartic: Cutoff for quartic force constants (Å).
        cutoff_distance_quintic: Cutoff for quintic force constants (Å).
        cutoff_distance_sextic: Cutoff for sextic force constants (Å).
        no_lasso_fitting_anhar: Disable LASSO for the anharmonic fit.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        atom_disp: float = 0.015,
        atom_disp_anhar: float = 0.1,
        supercell_matrix: ArrayLike | None = None,
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
        fitting_method: str = "LASSO",
        num_harmonic_snapshots: int | None = None,
        num_anharmonic_snapshots: int | None = None,
        calc_anharmonic: bool = False,
        symprec: float = 1e-5,
        min_length: float = 12.0,
        force_diagonal: bool = True,
        cutoff_distance_cubic: float = 6.3,
        cutoff_distance_quartic: float = 5.3,
        cutoff_distance_quintic: float = 3.3,
        cutoff_distance_sextic: float = 3.3,
        no_lasso_fitting_anhar: bool = False,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            atom_disp: Harmonic finite-difference displacement (Å).
            atom_disp_anhar: Anharmonic finite-difference displacement (Å).
            supercell_matrix: 3x3 supercell matrix; if None, derived from
                ``min_length`` via ``CubicSupercellTransformation``.
            t_step: Temperature step for thermal properties (K).
            t_max: Maximum temperature for thermal properties (K).
            t_min: Minimum temperature for thermal properties (K).
            fmax: Relaxation force tolerance (eV/Å).
            optimizer: Optimizer name for pre-relaxation.
            relax_structure: Whether to relax before phonon calculation.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
            write_force_constants: Output path for force constants, or False.
            write_band_structure: Output path for band structure YAML, or
                False.
            write_total_dos: Output path for total DOS data, or False.
            write_phonon: Output path for phonopy save file, or False.
            fitting_method: One of ``"FDM"``, ``"LASSO"``, ``"MD"``.
            num_harmonic_snapshots: Snapshots for harmonic fitting; defaults
                to 2x the number of generated displacements.
            num_anharmonic_snapshots: Snapshots for anharmonic fitting;
                defaults to 10x the number of generated displacements.
            calc_anharmonic: Whether to compute anharmonic force constants.
            symprec: Symmetry precision for spglib.
            min_length: Minimum supercell length (Å).
            force_diagonal: Force a diagonal supercell transformation.
            cutoff_distance_cubic: Cutoff for cubic force constants (Å).
            cutoff_distance_quartic: Cutoff for quartic force constants (Å).
            cutoff_distance_quintic: Cutoff for quintic force constants (Å).
            cutoff_distance_sextic: Cutoff for sextic force constants (Å).
            no_lasso_fitting_anhar: Disable LASSO for the anharmonic fit.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.atom_disp = atom_disp
        self.atom_disp_anhar = atom_disp_anhar
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
        self.fitting_method = fitting_method
        self.num_harmonic_snapshots = num_harmonic_snapshots
        self.num_anharmonic_snapshots = num_anharmonic_snapshots
        self.calc_anharmonic = calc_anharmonic
        self.symprec = symprec
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.cutoff_distance_cubic = cutoff_distance_cubic
        self.cutoff_distance_quartic = cutoff_distance_quartic
        self.cutoff_distance_quintic = cutoff_distance_quintic
        self.cutoff_distance_sextic = cutoff_distance_sextic
        self.no_lasso_fitting_anhar = no_lasso_fitting_anhar

        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:  # noqa: C901, PLR0912, PLR0915
        """Compute phonon and thermal properties via pheasy.

        Args:
            structure: Pymatgen structure or a dict containing
                ``final_structure``.

        Returns:
            A dict containing the input result merged with:

            - ``phonon``: ``phonopy.Phonopy`` with force constants populated.
            - ``thermal_properties``: per-temperature free energy, entropy,
              and heat capacity (units as documented by phonopy).
        """
        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator, fmax=self.fmax, optimizer=self.optimizer, **(self.relax_calc_kwargs or {})
            )
            result |= relaxer.calc(structure_in)
            structure_in = result["final_structure"]

        cell = get_phonopy_structure(structure_in)

        if self.supercell_matrix is None:
            transformation = CubicSupercellTransformation(
                min_length=self.min_length, force_diagonal=self.force_diagonal
            )
            transformation.apply_transformation(structure_in)
            self.supercell_matrix = np.array(transformation.transformation_matrix.transpose().tolist())

        phonon = phonopy.Phonopy(cell, self.supercell_matrix)  # type: ignore[arg-type]

        if self.fitting_method == "FDM":
            phonon.generate_displacements(distance=self.atom_disp)
        elif self.fitting_method == "LASSO":
            if self.num_harmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp)
                self.num_harmonic_snapshots = len(phonon.displacements) * 2
                phonon.generate_displacements(
                    distance=self.atom_disp, number_of_snapshots=self.num_harmonic_snapshots, random_seed=42
                )
        elif self.fitting_method == "MD":
            logger.info("MD fitting method is not implemented yet.")
        else:
            raise ValueError(f"Unknown fitting method: {self.fitting_method}")

        disp_supercells = phonon.supercells_with_displacements

        phonon.forces = [  # type: ignore[assignment]
            _calc_forces(self.calculator, supercell)
            for supercell in disp_supercells  # type: ignore[union-attr]
            if supercell is not None
        ]

        force_equilibrium = _calc_forces(self.calculator, phonon.supercell)  # type: ignore[union-attr]
        phonon.forces = np.array(phonon.forces) - force_equilibrium  # type: ignore[assignment]

        disp_array = np.array(
            [supercell.get_positions() - phonon.supercell.get_positions() for supercell in disp_supercells]
        )

        logger.info("Forces computed for supercells; producing force constants.")

        with open("disp_matrix.pkl", "wb") as file:
            pickle.dump(disp_array, file)
        with open("force_matrix.pkl", "wb") as file:
            pickle.dump(phonon.forces, file)

        supercell = phonon.get_supercell()

        logger.info("Writing POSCAR and SPOSCAR for pheasy.")
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        num_har = disp_array.shape[0]
        supercell_matrix = np.asarray(self.supercell_matrix)

        logger.info(
            "Running pheasy for second-order force constants. "
            "Please cite: Lin, Poncé, Marzari, npj Comput. Mater. 8, 236 (2022)."
        )

        dim = (int(supercell_matrix[0][0]), int(supercell_matrix[1][1]), int(supercell_matrix[2][2]))

        pheasy_cmd_1 = (
            f'pheasy --dim "{dim[0]}" "{dim[1]}" "{dim[2]}" -s -w 2 --symprec "{float(self.symprec)}" --nbody 2'
        )
        # Build the null-space basis to reduce free parameters and enforce
        # physical constraints on the harmonic FCs.
        pheasy_cmd_2 = f'pheasy --dim "{dim[0]}" "{dim[1]}" "{dim[2]}" -c --symprec "{float(self.symprec)}" -w 2'
        # Build the compressive-sensing (displacement) matrix used as input
        # to the LASSO fit.
        pheasy_cmd_3 = (
            f'pheasy --dim "{dim[0]}" "{dim[1]}" "{dim[2]}" '
            f'-w 2 -d --symprec "{float(self.symprec)}" --ndata "{int(num_har)}" --disp_file'
        )
        pheasy_cmd_4 = (
            f'pheasy --dim "{dim[0]}" "{dim[1]}" "{dim[2]}" '
            f'-f --full_ifc -w 2 --symprec "{float(self.symprec)}" '
            f'--rasr BHH --ndata "{int(num_har)}"'
        )

        for cmd in (pheasy_cmd_1, pheasy_cmd_2, pheasy_cmd_3, pheasy_cmd_4):
            subprocess.call(cmd, shell=True)  # noqa: S602

        phonon.force_constants = parse_FORCE_CONSTANTS(filename="FORCE_CONSTANTS")
        phonon.symmetrize_force_constants()

        # FORCE_CONSTANTS_2ND is the harmonic FC file consumed by ShengBTE
        # and FourPhonon for thermal-conductivity calculations.
        write_force_constants(phonon.force_constants, filename="FORCE_CONSTANTS_2ND")

        logger.info("Harmonic force constants ready; running phonon calculations.")
        phonon.run_mesh()
        phonon.run_thermal_properties(t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)

        if self.write_force_constants:
            write_force_constants(phonon.force_constants, filename=self.write_force_constants)
        if self.write_band_structure:
            phonon.auto_band_structure(write_yaml=True, filename=self.write_band_structure)
        if self.write_total_dos:
            phonon.auto_total_dos(write_dat=True, filename=self.write_total_dos)
        if self.write_phonon:
            phonon.save(filename=self.write_phonon)

        if self.calc_anharmonic:
            logger.info("Calculating anharmonic force constants with LASSO in pheasy.")

            # Remove stale harmonic disp/force matrices before regenerating
            # anharmonic ones — pheasy will pick up whichever files exist.
            subprocess.call("rm -f disp_matrix.pkl force_matrix.pkl", shell=True)  # noqa: S602, S607

            if self.num_anharmonic_snapshots is None:
                phonon.generate_displacements(distance=self.atom_disp_anhar)
                self.num_anharmonic_snapshots = len(phonon.displacements) * 10
            phonon.generate_displacements(
                distance=self.atom_disp_anhar,
                number_of_snapshots=self.num_anharmonic_snapshots,
                random_seed=42,
            )
            disp_supercells = phonon.supercells_with_displacements
            phonon.forces = [  # type: ignore[assignment]
                _calc_forces(self.calculator, supercell)
                for supercell in disp_supercells  # type: ignore[union-attr]
                if supercell is not None
            ]
            force_equilibrium = _calc_forces(self.calculator, phonon.supercell)
            phonon.forces = np.array(phonon.forces) - force_equilibrium
            disp_array = np.array(
                [supercell.get_positions() - phonon.supercell.get_positions() for supercell in disp_supercells]
            )

            with open("disp_matrix.pkl", "wb") as file:
                pickle.dump(disp_array, file)
            with open("force_matrix.pkl", "wb") as file:
                pickle.dump(phonon.forces, file)

            num_anh = disp_array.shape[0]

            pheasy_cmd_5 = (
                f"pheasy --dim {dim[0]} {dim[1]} {dim[2]} -s -w 4 "
                f"--symprec {float(self.symprec)} --nbody 2 3 3 "
                f"--c3 {float(self.cutoff_distance_cubic)} --c4 {float(self.cutoff_distance_quartic)}"
            )
            pheasy_cmd_6 = f"pheasy --dim {dim[0]} {dim[1]} {dim[2]} -c --symprec {float(self.symprec)} -w 4"
            pheasy_cmd_7 = (
                f"pheasy --dim {dim[0]} {dim[1]} {dim[2]} -w 4 -d "
                f"--symprec {float(self.symprec)} --ndata {int(num_anh)} --disp_file"
            )
            pheasy_cmd_8 = (
                f"pheasy --dim {dim[0]} {dim[1]} {dim[2]} -f -w 4 --fix_fc2 "
                f"--symprec {float(self.symprec)} --ndata {int(num_anh)}"
            )
            if not self.no_lasso_fitting_anhar:
                pheasy_cmd_8 += " -l LASSO --std"

            for cmd in (pheasy_cmd_5, pheasy_cmd_6, pheasy_cmd_7, pheasy_cmd_8):
                logger.info("pheasy: %s", cmd)
                subprocess.call(cmd, shell=True)  # noqa: S602

            logger.info("Higher-order force constants ready.")

        return result | {"phonon": phonon, "thermal_properties": phonon.get_thermal_properties_dict()}


def _calc_forces(calculator: Calculator, supercell: PhonopyAtoms) -> ArrayLike:
    """Compute forces on a supercell using the given ASE calculator.

    Args:
        calculator: ASE calculator.
        supercell: Phonopy supercell.

    Returns:
        Forces array (N x 3) in eV/Å.
    """
    struct = get_pmg_structure(supercell)
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calculator
    return atoms.get_forces()
