"""Calculators for EOS and associated properties."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.eos import BirchMurnaghan
from sklearn.metrics import r2_score

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class EOSCalc(PropCalc):
    """
    Performs equation of state (EOS) calculations using a specified ASE calculator.

    This class is intended to fit the Birch-Murnaghan equation of state, determine the
    bulk modulus, and provide other relevant physical properties of a given structure.
    The EOS calculation includes applying volumetric strain to the structure, optional
    initial relaxation of the structure, and evaluation of energies and volumes
    corresponding to the applied strain.

    Attributes:
        calculator: ASE calculator or universal model name.
        optimizer: Optimizer for optional relaxations.
        relax_structure: Relax initial structure before the EOS scan.
        n_points: Number of volume/strain samples.
        max_abs_strain: Half-range of applied volumetric strain (symmetric scan).
        fmax: Force tolerance for relaxations.
        max_steps: Max optimizer steps per relaxation.
        allow_shape_change: Allow cell shape to change at fixed volume during EOS points.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        optimizer: Optimizer | str = "FIRE",
        max_steps: int = 500,
        max_abs_strain: float = 0.1,
        n_points: int = 11,
        fmax: float = 0.1,
        allow_shape_change: bool = True,
        relax_structure: bool = True,
        relax_calc_kwargs: dict | None = None,
        r2_min: float = 0.95,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            optimizer: ASE optimizer name or class for relaxations.
            max_steps: Maximum relaxation steps per structure.
            max_abs_strain: Maximum absolute volumetric strain in the scan.
            n_points: Number of strain points (including halves of the scan).
            fmax: Force convergence criterion (eV/Å).
            allow_shape_change: Relax cell shape at fixed volume for EOS points when True.
            relax_structure: Relax input structure before the strain scan.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
            r2_min: Minimum acceptable R² for the Birch-Murnaghan fit. A
                ``RuntimeWarning`` is emitted (and the value is still returned)
                if the fit's R² falls below this threshold. Set to a negative
                number to disable the check.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.max_abs_strain = max_abs_strain
        self.n_points = n_points
        self.fmax = fmax
        self.allow_shape_change = allow_shape_change
        self.relax_structure = relax_structure
        self.relax_calc_kwargs = relax_calc_kwargs
        self.r2_min = r2_min

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with ``eos`` (``volumes`` in A^3, ``energies`` in eV), ``bulk_modulus_bm``
            (GPa), ``r2_score_bm`` (dimensionless), and fields from the final relaxation
            merged in. ``_units`` maps each numeric output to its unit string. See pymatgen
            ``BirchMurnaghan`` / ``EOSBase`` for fit details.
        """
        result = super().calc(structure)
        relax_calc_kwargs = {
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "max_steps": self.max_steps,
            **(self.relax_calc_kwargs or {}),
        }
        result, raw_structure = self._check_and_prelax(
            result["final_structure"],
            result,
            **relax_calc_kwargs,
        )
        structure_in: Structure = to_pmg_structure(raw_structure)

        volumes, energies = [], []

        # Don't relax the volume!
        relax_calc_kwargs["relax_cell"] = bool(self.allow_shape_change)
        relax_calc_kwargs["cell_filter_kwargs"] = {"constant_volume": True} if self.allow_shape_change else {}
        relaxer = RelaxCalc(self.calculator, **relax_calc_kwargs)

        temp_structure = structure_in.copy()
        for idx in np.linspace(-self.max_abs_strain, self.max_abs_strain, self.n_points)[self.n_points // 2 :]:
            structure_strained = temp_structure.copy()
            structure_strained.apply_strain(
                (((1 + idx) ** 3 * structure_in.volume) / (structure_strained.volume)) ** (1 / 3) - 1
            )
            r = relaxer.calc(structure_strained)
            volumes.append(r["final_structure"].volume)
            energies.append(r["energy"])
            temp_structure = r["final_structure"]

        for idx in np.flip(np.linspace(-self.max_abs_strain, self.max_abs_strain, self.n_points)[: self.n_points // 2]):
            structure_strained = structure_in.copy()
            structure_strained.apply_strain(
                (((1 + idx) ** 3 * structure_in.volume) / (structure_strained.volume)) ** (1 / 3) - 1
            )
            r = relaxer.calc(structure_strained)
            volumes.append(r["final_structure"].volume)
            energies.append(r["energy"])

        bm = BirchMurnaghan(volumes=volumes, energies=energies)
        bm.fit()

        volumes, energies = map(
            list, zip(*sorted(zip(volumes, energies, strict=True), key=lambda i: i[0]), strict=False)
        )

        r2_bm = r2_score(energies, bm.func(volumes))
        if r2_bm < self.r2_min:
            warnings.warn(
                f"Birch-Murnaghan EOS fit R²={r2_bm:.4f} is below r2_min={self.r2_min}. "
                f"The bulk modulus may be unreliable; check that the energy-volume scan is "
                f"monotonic over [-{self.max_abs_strain}, {self.max_abs_strain}] and that "
                f"the calculator is converged.",
                RuntimeWarning,
                stacklevel=2,
            )

        eos_units = {**result.get("_units", {}), "eos.volumes": "A^3", "eos.energies": "eV", "bulk_modulus_bm": "GPa"}
        return result | {
            "eos": {"volumes": volumes, "energies": energies},
            "bulk_modulus_bm": bm.b0_GPa,
            "r2_score_bm": r2_bm,
            "_units": eos_units,
        }
