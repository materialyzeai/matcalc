"""Calculator for elastic properties."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core.elasticity import DeformedStructureSet, ElasticTensor, Strain, Stress
from pymatgen.core.elasticity.elastic import get_strain_state_dict

from ._base import PropCalc
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

    from ._relaxation import RelaxCalc


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
            r2_min: Minimum acceptable pooled R² of the per-component linear
                strain-stress fits (see ``r2_score_mean``). A ``RuntimeWarning`` is emitted (and values
                are still returned) when the pooled R² drops below this. Set
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
            ``bulk_modulus_vrh`` and ``youngs_modulus`` are returned in GPa if
            ``units_GPa=True``, otherwise in eV/A^3 (pymatgen's native units).
            ``residuals_sum`` (sum of squared stress residuals) is in the square of
            those units. ``r2_score_mean`` is the pooled R² of the strain-stress fits over the
            symmetry-allowed C_ij (see ``_elastic_tensor_from_strains``); the key name is historical,
            from when it was the mean of the per-component R² values.
            ``_units`` is a dict mapping each numeric output to its unit string.
        """
        result = super().calc(structure)
        structure_in = result["final_structure"]

        result, structure_in = self._check_and_prelax(structure_in, result, fmax=self.fmax)
        relax_calc: RelaxCalc | None = None
        if self.relax_deformed_structures:
            relax_calc = self._relaxer(fmax=self.fmax)
            relax_calc.relax_cell = False

        deformed_structure_set = DeformedStructureSet(
            to_pmg_structure(structure_in),
            self.norm_strains,
            self.shear_strains,
            self.symmetry,
        )
        strains = []
        stresses = []
        # DeformedStructureSet iterates its deformed structures in the order of ``deformations``.
        for deformation, deformed_structure in zip(
            deformed_structure_set.deformations, deformed_structure_set, strict=True
        ):
            if relax_calc is not None:
                # The relaxation result already carries the stress of the relaxed structure.
                stress = relax_calc.calc(deformed_structure)["stress"]
            else:
                stress = run_pes_calc(deformed_structure, self.calculator).stress
            strain = Strain.from_deformation(deformation)
            strains.append(strain)
            stresses.append(stress)
            if self.symmetry:
                # Regenerate the symmetrically equivalent deformations removed by the reduction so
                # that every independent strain state is available to the fit. The reduction matches
                # deformations to within 1e-8, so zero that round-off; otherwise a single-component
                # strain would be binned as a spurious mixed strain state.
                for symmop in deformed_structure_set.sym_dict[deformation]:
                    strains.append(strain.transform(symmop).zeroed(1e-6))
                    stresses.append(Stress(stress).transform(symmop))

        eq_stress = run_pes_calc(structure_in, self.calculator).stress if self.use_equilibrium else None
        elastic_tensor, residuals_sum, mean_r2 = self._elastic_tensor_from_strains(
            strains,
            stresses,
            eq_stress=eq_stress,
            structure=to_pmg_structure(structure_in),
        )
        if mean_r2 < self.r2_min:
            warnings.warn(
                f"Elastic strain-stress fits have pooled R²={mean_r2:.4f} below r2_min={self.r2_min}. "
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
            # Sum of squared stress residuals, hence squared stress units.
            "residuals_sum": "GPa^2" if self.units_GPa else "(eV/A^3)^2",
        }
        return result | {
            "elastic_tensor": elastic_tensor * factor,
            "shear_modulus_vrh": g * factor,
            "bulk_modulus_vrh": k * factor,
            "youngs_modulus": y * factor,
            "residuals_sum": residuals_sum * factor**2,
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
        structure: Structure | None = None,
    ) -> tuple[ElasticTensor, float, float]:
        """
        Fit elastic constants from strain-stress pairs (Voigt), optionally subtracting
        equilibrium stress.

        Args:
            strains: Strain states (array-like) for each deformation.
            stresses: Matching stress tensors (array-like).
            eq_stress: Equilibrium stress to subtract; None to omit.
            tol: Small components below this are zeroed on the fitted tensor.
            structure: The undeformed structure, whose point group sets which C_ij are allowed by
                symmetry. None scores all 36 components.

        Returns:
            ``(ElasticTensor, residuals_sum, mean_r2)``. ``residuals_sum`` is the total sum of
            squared stress residuals over the 36 per-component linear fits. ``mean_r2`` is the
            pooled coefficient of determination, ``1 - sum(SS_res) / sum(SS_tot)``, over the
            components allowed by the point group of ``structure`` (see ``_symmetry_allowed``).
            Components that vanish by symmetry are excluded: their R² only measures numerical
            noise or second-order (even-in-strain) response, which does not bias the fitted
            constants but would otherwise dominate the score. Small but symmetry-allowed constants
            (e.g. a near-zero C12) are kept, so a nonlinear response in them is still flagged.
        """
        strain_states = [tuple(ss) for ss in np.eye(6)]
        ss_dict = get_strain_state_dict(strains, stresses, eq_stress=eq_stress, add_eq=self.use_equilibrium)
        c_ij = np.zeros((6, 6))
        ss_res = np.zeros((6, 6))
        ss_tot = np.zeros((6, 6))
        for ii in range(6):
            strain = ss_dict[strain_states[ii]]["strains"]
            stress = ss_dict[strain_states[ii]]["stresses"]
            for jj in range(6):
                x = strain[:, ii]
                y = stress[:, jj]
                fit = np.polyfit(x, y, 1, full=True)
                c_ij[ii, jj] = fit[0][0]
                ss_res[ii, jj] = fit[1][0] if len(fit[1]) > 0 else 0.0
                ss_tot[ii, jj] = np.sum((y - np.mean(y)) ** 2)
        elastic_tensor = ElasticTensor.from_voigt(c_ij)
        residuals_sum = float(ss_res.sum())
        significant = _symmetry_allowed(structure) if structure is not None else np.ones((6, 6), dtype=bool)
        ss_tot_sum = float(ss_tot[significant].sum())
        mean_r2 = 1.0 - float(ss_res[significant].sum()) / ss_tot_sum if ss_tot_sum > 0 else 1.0
        return elastic_tensor.zeroed(tol), residuals_sum, mean_r2


def _symmetry_allowed(structure: Structure, symprec: float = 0.01) -> np.ndarray:
    """Mask of the Voigt C_ij that the point group of ``structure`` allows to be non-zero.

    A generic (random, symmetric) Voigt tensor is averaged over the point-group operations of the
    structure in its Cartesian frame, i.e. the frame in which the strains are applied; the components
    it zeroes are the ones forbidden by symmetry.

    Args:
        structure: The undeformed structure.
        symprec: Symmetry tolerance for ``SpacegroupAnalyzer``.

    Returns:
        Boolean (6, 6) array, True where C_ij is allowed by symmetry.
    """
    generic = np.random.default_rng(0).uniform(1, 2, (6, 6))
    voigt = ElasticTensor.from_voigt(generic + generic.T).fit_to_structure(structure, symprec=symprec).voigt
    return np.abs(voigt) > 1e-8 * np.abs(voigt).max()
