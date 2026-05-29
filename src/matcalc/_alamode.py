"""Calculator for phonon properties using alamode."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import phonopy
from ase.units import Bohr, Rydberg
from lxml import etree
from phonopy.file_IO import (
    parse_FORCE_CONSTANTS,
    write_FORCE_CONSTANTS,
)
from phonopy.harmonic.force_constants import (
    compact_fc_to_full_fc,
)
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

# Eight-decimal conversions used to write Alamode's DFSET in Bohr / Ry units.
BOHR_PER_ANGSTROM = 1.0 / Bohr
RYD_PER_EV_ANGSTROM = 1.0 / (Rydberg / Bohr)


class AlamodeCalc(PropCalc):
    """
    Phonon and higher-order force-constant calculations driven by Alamode.

    Generates harmonic (and optionally anharmonic) displacement-force datasets
    and invokes the Alamode ``alm`` solver to extract force constants, which
    are then written in phonopy ``FORCE_CONSTANTS`` format for downstream use.

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
        alm_prefix: Alamode ``PREFIX`` for output files.
        alm_command: Shell command used to invoke Alamode's ``alm`` solver.
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
        alm_prefix: str = "alamode",
        alm_command: str = "alm",
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
            alm_prefix: Alamode ``PREFIX`` for output files.
            alm_command: Shell command used to invoke Alamode's ``alm``
                solver (e.g. ``"alm"`` or ``"mpirun -n 1 /path/to/alm"``).
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
        self.alm_prefix = alm_prefix
        self.alm_command = alm_command

        for key, val, default_path in (
            ("write_force_constants", self.write_force_constants, "force_constants"),
            ("write_band_structure", self.write_band_structure, "band_structure.yaml"),
            ("write_total_dos", self.write_total_dos, "total_dos.dat"),
            ("write_phonon", self.write_phonon, "phonon.yaml"),
        ):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:  # noqa: PLR0915
        """Compute force constants via Alamode and return a phonopy object.

        Args:
            structure: Pymatgen structure or a dict containing
                ``final_structure``.

        Returns:
            The input result merged with ``{"phonon": Phonopy}`` populated
            with Alamode-derived force constants.
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

        _write_alamode_dfset("DFSET_harmonic", disp_array, phonon.forces)

        supercell = phonon.get_supercell()
        write_vasp("POSCAR", cell)
        write_vasp("SPOSCAR", supercell)

        scaling_factor, lattice_vectors, elements, _num_atoms, positions = _read_sposcar("SPOSCAR")

        # Write harmonic Alamode input file.
        _write_alamode_input(
            "alamode.in",
            prefix=self.alm_prefix,
            elements=elements,
            positions=positions,
            scaling_factor=scaling_factor,
            lattice_vectors=lattice_vectors,
            norder=1,
            cutoff_line="*-*  18",
            optimize_extra={"DFSET": "DFSET_harmonic"},
        )

        subprocess.run(f"{self.alm_command} alamode.in", shell=True, check=False)  # noqa: S602

        map_p2s, fc2_compact = _get_forceconstants_xml(f"{self.alm_prefix}.xml")
        _write_fc2_phonopy(map_p2s, fc2_compact, filename="FORCE_CONSTANTS")

        fc = parse_FORCE_CONSTANTS("FORCE_CONSTANTS")
        fc_full = compact_fc_to_full_fc(phonon.primitive, fc, log_level=0)
        write_FORCE_CONSTANTS(force_constants=fc_full, filename="FORCE_CONSTANTS_2ND")

        if self.calc_anharmonic:
            logger.info("Calculating anharmonic force constants with Alamode.")

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
            num_anh = disp_array.shape[0]

            _write_alamode_dfset("DFSET_anharmonic", disp_array, phonon.forces)

            _write_alamode_input(
                "alamode_anhar.in",
                prefix=self.alm_prefix,
                elements=elements,
                positions=positions,
                scaling_factor=scaling_factor,
                lattice_vectors=lattice_vectors,
                norder=3,
                cutoff_line="*-* 18 10 7.5",
                general_extra={"FC3_SHENGBTE": 1, "FC4_SHENGBTE": 1},
                interaction_extra={"NBODY": "2 3 3"},
                optimize_extra={
                    "NDATA": num_anh,
                    "LMODEL": "enet",
                    "DFSET": "DFSET_anharmonic",
                    "FC2XML": f"{self.alm_prefix}.xml",
                    "ICONST": 11,
                    "L1_ALPHA": 3.82925e-06,
                    "STANDARDIZE": 1,
                    "CONV_TOL": 1.0e-8,
                },
            )

            subprocess.run(f"{self.alm_command} alamode_anhar.in", shell=True, check=False)  # noqa: S602
            logger.info("Higher-order force constants ready.")

        return result | {"phonon": phonon}


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


def _write_alamode_dfset(path: str, disp_array: np.ndarray, forces: np.ndarray) -> None:
    """Write an Alamode ``DFSET`` (displacements in Bohr, forces in Ry/Bohr)."""
    with open(path, "w") as f:
        for i, (disp, force) in enumerate(zip(disp_array, forces, strict=False)):
            f.write(f"# supercell {i + 1}\n")
            for d, fr in zip(disp, force, strict=False):
                vals = np.concatenate((d * BOHR_PER_ANGSTROM, fr * RYD_PER_EV_ANGSTROM))
                f.write(" ".join(f"{v:.8f}" for v in vals) + "\n")


def _read_sposcar(path: str) -> tuple[float, list[list[float]], list[str], list[int], list[list[float]]]:
    """Parse the bits of a VASP SPOSCAR file needed for the Alamode input."""
    with open(path) as f:
        lines = f.readlines()
    scaling_factor = float(lines[1].strip())
    lattice_vectors = [list(map(float, lines[i].split())) for i in range(2, 5)]
    elements = lines[5].split()
    num_atoms = list(map(int, lines[6].split()))
    total_atoms = sum(num_atoms)
    position_lines = lines[8 : 8 + total_atoms]
    positions: list[list[float]] = []
    for element_idx, count in enumerate(num_atoms):
        for _ in range(count):
            coords = list(map(float, position_lines.pop(0).split()[:3]))
            positions.append([element_idx + 1, *coords])
    return scaling_factor, lattice_vectors, elements, num_atoms, positions


def _write_alamode_input(
    path: str,
    *,
    prefix: str,
    elements: list[str],
    positions: list[list[float]],
    scaling_factor: float,
    lattice_vectors: list[list[float]],
    norder: int,
    cutoff_line: str,
    general_extra: dict[str, Any] | None = None,
    interaction_extra: dict[str, Any] | None = None,
    optimize_extra: dict[str, Any] | None = None,
) -> None:
    """Write an Alamode ``alm`` input file with the standard namelist layout."""
    general_extra = general_extra or {}
    interaction_extra = interaction_extra or {}
    optimize_extra = optimize_extra or {}
    total_atoms = len(positions)

    with open(path, "w") as f:
        f.write("&general\n")
        f.write(f"  PREFIX = {prefix}\n")
        f.write("  MODE = optimize\n")
        f.write(f"  NAT = {total_atoms}\n")
        f.write(f"  NKD = {len(elements)}\n")
        f.write("  KD = " + " ".join(elements) + "\n")
        f.writelines(f"  {key} = {val}\n" for key, val in general_extra.items())
        f.write("/\n\n")

        f.write("&interaction\n")
        f.write(f"  NORDER = {norder}\n")
        f.writelines(f"  {key} = {val}\n" for key, val in interaction_extra.items())
        f.write("/\n\n")

        f.write("&cutoff\n")
        f.write(f"  {cutoff_line}\n")
        f.write("/\n\n")

        f.write("&optimize\n")
        f.writelines(f"  {key} = {val}\n" for key, val in optimize_extra.items())
        f.write("/\n\n")

        f.write("&cell\n")
        f.write(f"  {scaling_factor:.16f} # factor\n")
        f.writelines(f"  {vec[0]:.10f}   {vec[1]:.10f}   {vec[2]:.10f}\n" for vec in lattice_vectors)
        f.write("# cell matrix\n/\n\n")

        f.write("&position\n")
        f.writelines(f"  {int(pos[0]):d}   {pos[1]:.16f}   {pos[2]:.16f}   {pos[3]:.16f}\n" for pos in positions)
        f.write("/\n")


def _parse_xml(fname_xml: str) -> etree._ElementTree:
    try:
        return etree.parse(fname_xml)
    except etree.XMLSyntaxError:
        repair_parser = etree.XMLParser(recover=True)
        return etree.parse(fname_xml, parser=repair_parser)


def _get_forceconstants_xml(fname_xml: str) -> tuple[np.ndarray, np.ndarray]:
    xml = _parse_xml(fname_xml)
    root = xml.getroot()

    natom_super = int(root.find("Structure/NumberOfAtoms").text)
    ntrans = int(root.find("Symmetry/NumberOfTranslations").text)
    natom_prim = natom_super // ntrans

    map_p2s = np.zeros((ntrans, natom_prim), dtype=int)
    for elems in root.findall("Symmetry/Translations/map"):
        itran = int(elems.get("tran")) - 1
        iatom = int(elems.get("atom")) - 1
        map_p2s[itran, iatom] = int(elems.text) - 1

    fc2_compact = np.zeros((natom_prim, natom_super, 3, 3), dtype=float)
    for elems in root.findall("ForceConstants/HARMONIC/FC2"):
        atom1, xyz1 = [int(t) - 1 for t in elems.get("pair1").split()]
        atom2, xyz2, _icell2 = [int(t) - 1 for t in elems.get("pair2").split()]
        fc2_compact[atom1, atom2, xyz1, xyz2] += float(elems.text)

    fc2_compact *= Rydberg / (Bohr**2)
    return map_p2s, fc2_compact


def _write_fc2_phonopy(map_p2s: np.ndarray, fc2_compact: np.ndarray, filename: str = "FORCE_CONSTANTS") -> None:
    natom_prim, natom_super = fc2_compact.shape[:2]
    with open(filename, "w") as f:
        f.write(f"{natom_prim:5d} {natom_super:5d}\n")
        for i in range(natom_prim):
            for j in range(natom_super):
                f.write(f"{map_p2s[0, i] + 1:5d} {j + 1:5d}\n")
                for k in range(3):
                    f.writelines(f"{fc2_compact[i, j, k, m]:20.15f}" for m in range(3))
                    f.write("\n")
