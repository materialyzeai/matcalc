"""Calculator for elastic properties."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.core.elasticity.elastic import get_strain_state_dict

from ._base import PropCalc
from ._relaxation import RelaxCalc
from .backend import run_pes_calc
from .utils import to_pmg_structure

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike
    from pymatgen.core import Structure


class ElasticityCalc(PropCalc):
    """
    Elastic tensor and related moduli via strain-stress fitting with pymatgen.

    Attributes:
        calculator: ASE calculator (or universal model name).
        norm_strains: Normal strains applied in ``DeformedStructureSet``.
        shear_strains: Shear strains applied in ``DeformedStructureSet``.
        fmax: Force tolerance for optional relaxations.
        symmetry: Whether to reduce deformations by symmetry.
        relax_structure: Relax initial structure before deforming.
        relax_deformed_structures: Relax each deformed structure before stress.
        use_equilibrium: Include equilibrium stress in the fit when applicable.
        units_GPa: If True, report moduli in GPa instead of pymatgen's native eV/A^3.
        relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        norm_strains: Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01),
        shear_strains: Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06),
        fmax: float = 0.1,
        symmetry: bool = False,
        relax_structure: bool = True,
        relax_deformed_structures: bool = False,
        use_equilibrium: bool = True,
        units_GPa: bool = False,  # noqa: N803
        relax_calc_kwargs: dict | None = None,
        r2_min: float = 0.95,
    ) -> None:
        """
        Args:
            calculator: ASE calculator or universal model name string.
            norm_strains: Normal strains (non-empty, no zeros); scalar broadcast to one value.
            shear_strains: Shear strains (non-empty, no zeros); scalar allowed.
            fmax: Force tolerance for relaxations.
            symmetry: Pass-through to pymatgen ``DeformedStructureSet``.
            relax_structure: Relax parent structure before generating deformations.
            relax_deformed_structures: Relax each deformed structure before stress eval.
            use_equilibrium: Use equilibrium stress in fit; forced True if only one strain type.
            units_GPa: If True, return moduli (and elastic tensor / residuals) in GPa.
                Defaults to False, in which case values are returned in pymatgen's native
                units of eV/A^3.
            relax_calc_kwargs: Optional kwargs for ``RelaxCalc``.
            r2_min: Minimum acceptable mean R² across the per-component linear
                strain-stress fits. A ``RuntimeWarning`` is emitted (and values
                are still returned) when the mean R² drops below this. Set
                negative to disable. Default 0.95.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.norm_strains = tuple(np.array([1]) * np.asarray(norm_strains))
        self.shear_strains = tuple(np.array([1]) * np.asarray(shear_strains))
        if len(self.norm_strains) == 0:
            raise ValueError("norm_strains is empty")
        if len(self.shear_strains) == 0:
            raise ValueError("shear_strains is empty")
        if 0 in self.norm_strains or 0 in self.shear_strains:
            raise ValueError("strains must be non-zero")
        self.relax_structure = relax_structure
        self.relax_deformed_structures = relax_deformed_structures
        self.fmax = fmax
        self.symmetry = symmetry
        if len(self.norm_strains) > 1 and len(self.shear_strains) > 1:
            self.use_equilibrium = use_equilibrium
        else:
            self.use_equilibrium = True
        self.units_GPa = units_GPa
        self.relax_calc_kwargs = relax_calc_kwargs
        self.r2_min = r2_min

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            structure: Pymatgen structure, ASE atoms, or dict with structure keys.

        Returns:
            Dict including ``elastic_tensor``, ``shear_modulus_vrh``, ``bulk_modulus_vrh``,
            ``youngs_modulus``, ``residuals_sum``, ``structure``, ``_units``, and merged
            relaxation fields. ``elastic_tensor``, ``shear_modulus_vrh``,
            ``bulk_modulus_vrh``, ``youngs_modulus`` and ``residuals_sum`` are returned
            in GPa if ``units_GPa=True``, otherwise in eV/A^3 (pymatgen's native units).
            ``_units`` is a dict mapping each numeric output to its unit string.
        """
        result = super().calc(structure)
        structure_in = result["final_structure"]

        relax_calc: RelaxCalc | None = None
        if self.relax_structure or self.relax_deformed_structures:
            relax_calc = RelaxCalc(self.calculator, fmax=self.fmax, **(self.relax_calc_kwargs or {}))
        result, structure_in = self._prerelax(structure_in, result, relaxer=relax_calc)
        if self.relax_deformed_structures and relax_calc is not None:
            relax_calc.relax_cell = False

        deformed_structure_set = DeformedStructureSet(
            to_pmg_structure(structure_in),
            self.norm_strains,
            self.shear_strains,
            self.symmetry,
        )
        stresses = []
        for deformed_structure in deformed_structure_set:
            if self.relax_deformed_structures and relax_calc is not None:
                deformed_relaxed = relax_calc.calc(deformed_structure)["final_structure"]
                sim = run_pes_calc(deformed_relaxed, self.calculator)
            else:
                sim = run_pes_calc(deformed_structure, self.calculator)
            stresses.append(sim.stress)

        strains = [Strain.from_deformation(deformation) for deformation in deformed_structure_set.deformations]
        sim = run_pes_calc(structure_in, self.calculator)
        elastic_tensor, residuals_sum, mean_r2 = self._elastic_tensor_from_strains(
            strains,
            stresses,
            eq_stress=sim.stress if self.use_equilibrium else None,
        )
        if mean_r2 < self.r2_min:
            warnings.warn(
                f"Elastic strain-stress fits have mean R²={mean_r2:.4f} below r2_min={self.r2_min}. "
                f"The elastic tensor may be unreliable; consider smaller |strains|, more strain "
                f"points, or enabling relax_deformed_structures.",
                RuntimeWarning,
                stacklevel=2,
            )
        factor = 1 if not self.units_GPa else 1 / elastic_tensor.GPa_to_eV_A3
        # Compute Young's modulus from the same VRH averages used for K, G so it
        # is dimensionally consistent. pymatgen's ``ElasticTensor.y_mod`` hardcodes
        # a 9e9 GPa->Pa factor that assumes K/G are in GPa; with pymatgen's native
        # eV/A^3 storage that produces nonsense units (see Issue #85).
        k = elastic_tensor.k_vrh
        g = elastic_tensor.g_vrh
        y = 9 * k * g / (3 * k + g)
        unit_str = "GPa" if self.units_GPa else "eV/A^3"
        units_map = {
            **result.get("_units", {}),
            "elastic_tensor": unit_str,
            "bulk_modulus_vrh": unit_str,
            "shear_modulus_vrh": unit_str,
            "youngs_modulus": unit_str,
            "residuals_sum": unit_str,
        }
        return result | {
            "elastic_tensor": elastic_tensor * factor,
            "shear_modulus_vrh": g * factor,
            "bulk_modulus_vrh": k * factor,
            "youngs_modulus": y * factor,
            "residuals_sum": residuals_sum * factor,
            "r2_score_mean": mean_r2,
            "structure": structure_in,
            "_units": units_map,
        }

    def _elastic_tensor_from_strains(
        self,
        strains: ArrayLike,
        stresses: ArrayLike,
        eq_stress: ArrayLike = None,
        tol: float = 1e-7,
    ) -> tuple[ElasticTensor, float, float]:
        """
        Fit elastic constants from strain-stress pairs (Voigt), optionally subtracting
        equilibrium stress.

        Args:
            strains: Strain states (array-like) for each deformation.
            stresses: Matching stress tensors (array-like).
            eq_stress: Equilibrium stress to subtract; None to omit.
            tol: Small components below this are zeroed on the fitted tensor.

        Returns:
            ``(ElasticTensor, residuals_sum, mean_r2)``. ``mean_r2`` is the average
            coefficient of determination across the 36 per-component linear fits;
            stress components with zero variance (zeroed by symmetry) are skipped.
        """
        strain_states = [tuple(ss) for ss in np.eye(6)]
        ss_dict = get_strain_state_dict(strains, stresses, eq_stress=eq_stress, add_eq=self.use_equilibrium)
        c_ij = np.zeros((6, 6))
        residuals_sum = 0.0
        r2_values: list[float] = []
        for ii in range(6):
            strain = ss_dict[strain_states[ii]]["strains"]
            stress = ss_dict[strain_states[ii]]["stresses"]
            for jj in range(6):
                x = strain[:, ii]
                y = stress[:, jj]
                fit = np.polyfit(x, y, 1, full=True)
                c_ij[ii, jj] = fit[0][0]
                residuals_sum += fit[1][0] if len(fit[1]) > 0 else 0.0
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                if ss_tot > 0:  # skip flat (zero by symmetry) components
                    ss_res = float(fit[1][0]) if len(fit[1]) > 0 else 0.0
                    r2_values.append(1.0 - ss_res / ss_tot)
        elastic_tensor = ElasticTensor.from_voigt(c_ij)
        mean_r2 = float(np.mean(r2_values)) if r2_values else 1.0
        return elastic_tensor.zeroed(tol), residuals_sum, mean_r2
