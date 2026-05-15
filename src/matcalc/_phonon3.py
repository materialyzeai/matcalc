"""Calculator for phonon-phonon interaction and related properties."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from phono3py import Phono3py
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from ._base import PropCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class Phonon3Calc(PropCalc):
    """
    Class for calculating thermal conductivity using third-order force constants.

    This class integrates with the Phono3py library to compute thermal conductivity
    based on third-order force constants (FC3). It includes capabilities for optional
    structure relaxation, displacement generation, and force calculations on
    supercells. Results include the thermal conductivity as a function of temperature
    and other intermediate configurations used in the calculation.

    Attributes:
        calculator: ASE calculator or universal model name.
        fc2_supercell: Supercell matrix for FC2 (phonon supercell).
        fc3_supercell: Supercell matrix for FC3.
        mesh_numbers: q-mesh dimensions for phono3py.
        disp_kwargs: Passed to ``generate_displacements``.
        thermal_conductivity_kwargs: Passed to ``run_thermal_conductivity``.
        relax_structure: Relax primitive structure before phono3py.
        relax_calc_kwargs: Kwargs for ``RelaxCalc``.
        fmax: Relaxation force tolerance (eV/Å).
        optimizer: Relaxation optimizer name.
        t_min, t_max, t_step: Temperature grid for κ(T) (K).
        write_phonon3: Path to save phono3py state, True for default, or False.
        write_kappa: Whether phono3py writes κ output files.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        fc2_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fc3_supercell: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        mesh_numbers: ArrayLike = (20, 20, 20),
        disp_kwargs: dict[str, Any] | None = None,
        thermal_conductivity_kwargs: dict | None = None,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        fmax: float = 0.1,
        optimizer: str = "FIRE",
        t_min: float = 0,
        t_max: float = 1000,
        t_step: float = 10,
        write_phonon3: bool | str | Path = False,
        write_kappa: bool = False,
    ) -> None:
        """
        Initializes the class for thermal conductivity calculation and structure relaxation
        utilizing third-order force constants (fc3). The class provides configurable
        parameters for the relaxation process, supercell settings, thermal conductivity
        calculation, and file output management.

        Args:
            calculator: ASE calculator or universal model name string.
            fc2_supercell: Phonon (FC2) supercell matrix.
            fc3_supercell: FC3 supercell matrix.
            mesh_numbers: q-point mesh shape.
            disp_kwargs: Kwargs for ``Phono3py.generate_displacements``.
            thermal_conductivity_kwargs: Kwargs for ``run_thermal_conductivity``.
            relax_structure: Relax input before building phono3py.
            relax_calc_kwargs: Kwargs for ``RelaxCalc``.
            fmax: Force tolerance for relaxation (eV/Å).
            optimizer: ASE optimizer for relaxation.
            t_min: Minimum temperature for κ (K).
            t_max: Maximum temperature for κ (K).
            t_step: Temperature step (K).
            write_phonon3: Save path, True for default YAML, or False.
            write_kappa: Pass ``write_kappa`` to phono3py RTA.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.fc2_supercell = fc2_supercell
        self.fc3_supercell = fc3_supercell
        self.mesh_numbers = mesh_numbers
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
        self.write_phonon3 = write_phonon3
        self.write_kappa = write_kappa

        # Set default paths for saving output files.
        for key, val, default_path in (("write_phonon3", self.write_phonon3, "phonon3.yaml"),):
            setattr(self, key, str({True: default_path, False: ""}.get(val, val)))  # type: ignore[arg-type]

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Performs thermal conductivity calculations using the Phono3py library.

        This method processes a given atomic structure and calculates its thermal
        conductivity through third-order force constants (FC3) computations. The
        process involves optional relaxation of the input structure, generation of
        displacements, and force calculations corresponding to the supercell
        structures. The results include computed thermal conductivity over specified
        temperatures, along with intermediate Phono3py configurations.

        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with ``phonon3`` (``Phono3py``), ``temperatures`` (K), and
            ``thermal_conductivity`` (spatially averaged kappa in W/(m*K) where defined;
            NaN if unavailable). ``_units`` maps each numeric output to its unit string.
            See phono3py RTA documentation for details.
        """
        result = super().calc(structure)
        result, structure_in = self._prerelax(
            result["final_structure"],
            result,
            fmax=self.fmax,
            optimizer=self.optimizer,
        )

        cell = get_phonopy_structure(to_pmg_structure(structure_in))
        phonon3 = Phono3py(
            unitcell=cell,
            supercell_matrix=self.fc3_supercell,
            phonon_supercell_matrix=self.fc2_supercell,
            primitive_matrix="auto",
        )  # type: ignore[arg-type]

        if self.mesh_numbers is not None:
            phonon3.mesh_numbers = self.mesh_numbers

        phonon3.generate_displacements(**self.disp_kwargs)

        phonon_forces = []
        for supercell in phonon3.phonon_supercells_with_displacements:
            struct_supercell = get_pmg_structure(supercell)
            r = run_pes_calc(struct_supercell, self.calculator)
            f = r.forces
            phonon_forces.append(f)
        fc2_set = np.array(phonon_forces)
        phonon3.phonon_forces = fc2_set

        forces = []
        for supercell in phonon3.supercells_with_displacements:
            if supercell is not None:
                struct_supercell = get_pmg_structure(supercell)
                r = run_pes_calc(struct_supercell, self.calculator)
                f = r.forces
                forces.append(f)
        fc3_set = np.array(forces)
        phonon3.forces = fc3_set

        phonon3.produce_fc2(symmetrize_fc2=True)
        phonon3.produce_fc3(symmetrize_fc3r=True)
        phonon3.init_phph_interaction()

        temperatures = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        phonon3.run_thermal_conductivity(
            temperatures=temperatures,
            **self.thermal_conductivity_kwargs,
            write_kappa=self.write_kappa,
        )

        kappa = np.asarray(phonon3.thermal_conductivity.kappa_TOT_RTA)
        if kappa.size == 0 or np.any(np.isnan(kappa)):
            warnings.warn(
                f"Phono3py thermal conductivity returned NaN or empty values for "
                f"{self.mesh_numbers=}, {self.fc3_supercell=}, T-range "
                f"[{self.t_min}, {self.t_max}]. Check the mesh density and supercell "
                f"size; very coarse meshes or very small supercells often produce NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            kappa_ave = np.nan
        else:
            kappa_ave = kappa[..., :3].mean(axis=-1)

        if self.write_phonon3:
            phonon3.save(filename=self.write_phonon3)

        return result | {
            "phonon3": phonon3,
            "temperatures": temperatures,
            "thermal_conductivity": np.squeeze(kappa_ave),
            "_units": self._merge_units(
                result,
                {
                    "temperatures": "K",
                    "thermal_conductivity": "W/(m*K)",
                },
            ),
        }
