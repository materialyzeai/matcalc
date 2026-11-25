"""Calculators for EOS and associated properties."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.eos import EOS, EOSError
from sklearn.metrics import r2_score

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

logger = logging.getLogger(__name__)

EOS_MODELS = ("birch", "birch_murnaghan", "murnaghan", "pourier_tarantola", "vinet")

if TYPE_CHECKING:
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer
    from pymatgen.core import Structure


class EOSCalc(PropCalc):
    """
    Performs equation of state (EOS) calculations using a specified ASE calculator.

    This class is intended to fit an equation of state (Birch-Murnaghan by default), determine
    the bulk modulus, and provide other relevant physical properties of a given structure.
    The EOS calculation includes applying volumetric strain to the structure, optional
    initial relaxation of the structure, and evaluation of energies and volumes
    corresponding to the applied strain.

    Attributes:
        calculator: ASE calculator or universal model name.
        optimizer: Optimizer for optional relaxations.
        relax_structure: Relax initial structure before the EOS scan.
        n_points: Number of volume/strain samples.
        max_abs_strain: Half-range of applied linear (isotropic) strain (symmetric scan).
        fmax: Force tolerance for relaxations.
        max_steps: Max optimizer steps per relaxation.
        allow_shape_change: Allow cell shape to change at fixed volume during EOS points.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
        eos: EOS model for the energy-volume fit (``birch``, ``birch_murnaghan``, ``murnaghan``,
            ``pourier_tarantola``, ``vinet``).
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
        eos: str = "birch_murnaghan",
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            optimizer: ASE optimizer name or class for relaxations.
            max_steps: Maximum relaxation steps per structure.
            max_abs_strain: Maximum absolute linear (isotropic) strain in the scan; the volume
                spans ``(1 ± max_abs_strain)**3`` times the reference volume.
            n_points: Number of strain points (including halves of the scan).
            fmax: Force convergence criterion (eV/Å).
            allow_shape_change: Relax cell shape at fixed volume for EOS points when True.
            relax_structure: Relax input structure before the strain scan.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
            r2_min: Minimum acceptable R² for the EOS fit. A
                ``RuntimeWarning`` is emitted (and the value is still returned)
                if the fit's R² falls below this threshold. Set to a negative
                number to disable the check.
            eos: EOS model for the energy-volume fit, one of the pymatgen ``EOS`` model names
                ``birch``, ``birch_murnaghan``, ``murnaghan``, ``pourier_tarantola`` and ``vinet``.

        Raises:
            ValueError: If ``eos`` is not one of the supported EOS models.
        """
        if eos not in EOS_MODELS:
            raise ValueError(f"Unknown eos {eos!r}. Allowed values: {EOS_MODELS}.")
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
        self.eos = eos

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict:
        """
        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict with ``eos`` (``volumes`` in A^3, ``energies`` in eV), ``bulk_modulus_eos``
            (GPa), ``r2_score_eos`` (dimensionless) of the fitted EOS model, and fields from the
            final relaxation merged in. ``_units`` maps each
            numeric output to its unit string. See pymatgen ``EOS`` / ``EOSBase`` for fit
            details. If the fit fails (pymatgen ``EOSError``), a ``RuntimeWarning`` is emitted
            and the bulk modulus and R² are NaN; ``eos`` is still returned.
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

        # Don't relax the volume! Keep any user cell-filter settings (e.g. ``mask``) on top of that.
        relax_calc_kwargs["relax_cell"] = bool(self.allow_shape_change)
        relax_calc_kwargs["cell_filter_kwargs"] = (
            {**(relax_calc_kwargs.get("cell_filter_kwargs") or {}), "constant_volume": True}
            if self.allow_shape_change
            else {}
        )
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

        eos_name = self.eos.replace("_", "-").title()
        eos_fit = EOS.MODELS[self.eos](volumes=volumes, energies=energies)
        try:
            eos_fit.fit()
        except EOSError as exc:
            warnings.warn(
                f"{eos_name} EOS fit failed ({exc}); bulk_modulus_eos and r2_score_eos are set to NaN. "
                f"The energy-volume data are returned in eos.",
                RuntimeWarning,
                stacklevel=2,
            )
            fitted = False
        else:
            fitted = True

        volumes, energies = map(
            list, zip(*sorted(zip(volumes, energies, strict=True), key=lambda i: i[0]), strict=False)
        )

        bulk_modulus = eos_fit.b0_GPa if fitted else float("nan")
        r2 = r2_score(energies, eos_fit.func(volumes)) if fitted else float("nan")
        if r2 < self.r2_min:
            warnings.warn(
                f"{eos_name} EOS fit R²={r2:.4f} is below r2_min={self.r2_min}. "
                f"The bulk modulus may be unreliable; check that the energy-volume scan is "
                f"monotonic over [-{self.max_abs_strain}, {self.max_abs_strain}] and that "
                f"the calculator is converged.",
                RuntimeWarning,
                stacklevel=2,
            )

        eos_units = {**result.get("_units", {}), "eos.volumes": "A^3", "eos.energies": "eV", "bulk_modulus_eos": "GPa"}
        return result | {
            "eos": {"volumes": volumes, "energies": energies},
            "bulk_modulus_eos": bulk_modulus,
            "r2_score_eos": r2,
            "_units": eos_units,
        }
