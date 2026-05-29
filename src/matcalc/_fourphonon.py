"""Calculator for phonon-phonon interaction and related properties using FOURPHONON and FCs from Pheasy."""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import numpy as np
from phonopy.interface.vasp import read_vasp, write_vasp
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.io.vasp import Kpoints
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
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class FourPhononCalc(PropCalc):
    """
    Thermal-conductivity calculator using third- and fourth-order FCs via ShengBTE/FourPhonon.

    Builds a ShengBTE/FourPhonon CONTROL file from the (optionally relaxed)
    input structure, invokes the external solver, and returns the parsed κ(T).
    Requires the ShengBTE/FourPhonon binaries on PATH (or paths supplied via
    ``path_to_shengbte`` / ``path_to_fourphonon``) and ``f90nml`` for the
    CONTROL-file writer.

    Attributes:
        calculator: ASE calculator or universal model name.
        min_length: Minimum supercell length (Å).
        force_diagonal: Force a diagonal supercell transformation.
        supercell_matrix: 3x3 supercell matrix; derived from ``min_length``
            when None.
        mesh_numbers: q-mesh dimensions for ShengBTE/FourPhonon.
        reciprocal_density: If set, derives ``mesh_numbers`` from the
            structure (suggested ~40000).
        disp_kwargs: Kwargs forwarded to displacement generation.
        thermal_conductivity_kwargs: Kwargs forwarded to κ post-processing.
        relax_structure: Relax structure before computation.
        relax_calc_kwargs: Kwargs for ``RelaxCalc``.
        fmax: Relaxation force tolerance (eV/Å).
        optimizer: ASE optimizer name for pre-relaxation.
        t_min, t_max, t_step: Temperature grid for κ(T) (K).
        t_single: Single temperature for non-sweep mode (K).
        write_phonon3: Output path for phono3py-style state, or False.
        write_kappa: Whether to write κ output files.
        calc_4ph: Include fourth-order (4ph) scattering.
        many_t: Sweep over multiple temperatures vs single ``t_single``.
        scalebroad: Gaussian broadening factor.
        core_number: MPI ranks for parallel execution.
        srun: Use ``srun`` for launching.
        mpirun: Use ``mpirun`` for launching.
        parallel_calc: Run solver in parallel mode.
        path_to_shengbte: Path to the ShengBTE executable (or None for PATH).
        path_to_fourphonon: Path to the FourPhonon executable (or None for
            PATH).
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        min_length: float = 12,
        force_diagonal: bool = True,
        supercell_matrix: ArrayLike | None = None,
        mesh_numbers: ArrayLike = (10, 10, 10),
        reciprocal_density: int | None = None,
        disp_kwargs: dict[str, Any] | None = None,
        thermal_conductivity_kwargs: dict | None = None,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 100,
        t_single: float = 300,
        write_phonon3: bool | str | Path = False,
        write_kappa: bool = False,
        calc_4ph: bool = True,
        many_t: bool = False,
        scalebroad: float = 0.1,
        core_number: int = 4,
        srun: bool = False,
        mpirun: bool = False,
        parallel_calc: bool = False,
        path_to_shengbte: str | None = None,
        path_to_fourphonon: str | None = None,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            min_length: Minimum supercell length (Å).
            force_diagonal: Force a diagonal supercell transformation.
            supercell_matrix: 3x3 supercell matrix; if None, derived from
                ``min_length``.
            mesh_numbers: q-mesh dimensions.
            reciprocal_density: Optional reciprocal density (suggested
                ~40000); overrides ``mesh_numbers`` when set.
            disp_kwargs: Kwargs forwarded to displacement generation.
            thermal_conductivity_kwargs: Kwargs forwarded to κ
                post-processing.
            relax_structure: Whether to relax before computation.
            relax_calc_kwargs: Kwargs for ``RelaxCalc``.
            fmax: Relaxation force tolerance (eV/Å).
            optimizer: ASE optimizer name for pre-relaxation.
            t_min: Minimum temperature for κ(T) sweep (K).
            t_max: Maximum temperature for κ(T) sweep (K).
            t_step: Temperature step for κ(T) sweep (K).
            t_single: Single temperature for non-sweep mode (K).
            write_phonon3: Output path for phono3py-style state, or False.
            write_kappa: Whether to write κ output files.
            calc_4ph: Include fourth-order (4ph) scattering.
            many_t: Sweep over multiple temperatures vs single ``t_single``.
            scalebroad: Gaussian broadening factor.
            core_number: MPI ranks for parallel execution.
            srun: Use ``srun`` for launching.
            mpirun: Use ``mpirun`` for launching.
            parallel_calc: Run solver in parallel mode.
            path_to_shengbte: Path to the ShengBTE executable (or None to
                use PATH).
            path_to_fourphonon: Path to the FourPhonon executable (or None
                to use PATH).
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.supercell_matrix = supercell_matrix
        self.mesh_numbers = mesh_numbers
        self.reciprocal_density = reciprocal_density
        self.disp_kwargs = disp_kwargs if disp_kwargs is not None else {}
        self.thermal_conductivity_kwargs = (
            thermal_conductivity_kwargs if thermal_conductivity_kwargs is not None else {}
        )
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs if relax_calc_kwargs is not None else {}
        self.fmax = fmax
        self.optimizer = optimizer
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.t_single = t_single
        self.write_phonon3 = write_phonon3
        self.write_kappa = write_kappa
        self.calc_4ph = calc_4ph
        self.many_t = many_t
        self.scalebroad = scalebroad
        self.core_number = core_number
        self.srun = srun
        self.mpirun = mpirun
        self.parallel_calc = parallel_calc
        self.path_to_shengbte = path_to_shengbte
        self.path_to_fourphonon = path_to_fourphonon

        for key, val, default_path in (("write_phonon3", self.write_phonon3, "phonon3.yaml"),):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | dict[str, Any]) -> dict:  # noqa: C901, PLR0912, PLR0915
        """Compute thermal conductivity via ShengBTE/FourPhonon.

        Args:
            structure: Pymatgen structure or a dict containing
                ``final_structure``.

        Returns:
            A dict with ``phonon3``, ``temperatures``, and
            ``thermal_conductivity``. The current implementation writes the
            CONTROL file and runs ShengBTE; κ parsing is not yet wired up
            and the latter two fields are placeholders.
        """
        try:
            import f90nml
        except ImportError as exc:
            raise ImportError("f90nml is required for FourPhononCalc; install with `pip install f90nml`.") from exc

        result = super().calc(structure)
        structure_in: Structure = result["final_structure"]

        if self.relax_structure:
            relaxer = RelaxCalc(
                self.calculator,
                fmax=self.fmax,
                optimizer=self.optimizer,
                **(self.relax_calc_kwargs or {}),
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

        supercell_matrix = np.asarray(self.supercell_matrix)

        write_vasp("POSCAR", cell)
        primitive_cell = read_vasp("POSCAR")
        symbols = primitive_cell.get_chemical_symbols()
        lattvec = primitive_cell.get_cell().T.tolist()  # column-wise (Fortran)
        positions = primitive_cell.get_scaled_positions().tolist()

        scell = [supercell_matrix[0][0], supercell_matrix[1][1], supercell_matrix[2][2]]
        unique_elements = list(dict.fromkeys(symbols))
        mesh_numbers = np.asarray(self.mesh_numbers)
        ngrid = [mesh_numbers[0], mesh_numbers[1], mesh_numbers[2]]

        if self.reciprocal_density:
            kpoints = Kpoints.automatic_density(structure=structure_in, kppa=self.reciprocal_density)
            ngrid = list(kpoints.kpts[0])

        allocations = {
            "nelements": len(unique_elements),
            "natoms": len(positions),
            "ngrid(:)": ngrid,
            "norientations": 0,
        }

        crystal = {
            "lfactor": 0.1,
            "lattvec": lattvec,
            "elements": unique_elements,
            "types": [unique_elements.index(el) + 1 for el in symbols],
            "positions": np.array(positions).tolist(),
            "scell": scell,
        }

        if self.many_t:
            parameters = {
                "T_min": self.t_min,
                "T_max": self.t_max,
                "T_step": self.t_step,
                "scalebroad": self.scalebroad,
            }
        else:
            parameters = {"T": self.t_single, "scalebroad": self.scalebroad}
        if self.calc_4ph:
            parameters |= {
                "num_sample_process_3ph_phase_space": -1,
                "num_sample_process_3ph": -1,
                "num_sample_process_4ph_phase_space": 100000,
                "num_sample_process_4ph": 100000,
            }

        flags = {"nonanalytic": False, "convergence": True, "nanowires": False}
        if self.calc_4ph:
            flags = {"four_phonon": True, **flags}

        ordered_namelists = f90nml.Namelist(
            [
                ("allocations", allocations),
                ("crystal", crystal),
                ("parameters", parameters),
                ("flags", flags),
            ]
        )
        ordered_namelists.write("CONTROL", force=True)

        logger.info("CONTROL file written; running ShengBTE/FourPhonon.")

        shengbte_exe = self.path_to_shengbte or "ShengBTE"
        try:
            if self.parallel_calc and self.srun:
                subprocess.run(["srun", "-n", str(self.core_number), shengbte_exe], check=True)  # noqa: S603, S607
            elif self.parallel_calc and self.mpirun:
                subprocess.run(["mpirun", "-np", str(self.core_number), shengbte_exe], check=True)  # noqa: S603, S607
            else:
                subprocess.run([shengbte_exe], check=True)  # noqa: S603
            logger.info("ShengBTE executed successfully.")
        except subprocess.CalledProcessError as exc:
            logger.exception("ShengBTE failed.")
            raise RuntimeError("Failed to execute ShengBTE. Check input files and parameters.") from exc
        except FileNotFoundError as exc:
            logger.exception("ShengBTE executable not found.")
            raise RuntimeError(
                "ShengBTE executable not found. Install it and ensure it is on PATH or pass `path_to_shengbte`."
            ) from exc

        # kappa.dat parsing is not yet wired up; the placeholder return
        # below keeps the calc-chain shape stable until it lands.
        return {
            "phonon3": None,
            "temperatures": None,
            "thermal_conductivity": np.squeeze(np.full((1, 1), np.nan)),
        }
