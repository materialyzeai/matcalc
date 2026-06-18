"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter

from matcalc import ElasticityCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize("relax_deformed_structures", [False, True])
def test_elastic_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    relax_deformed_structures: bool,
) -> None:
    """Tests for ElasticCalc class"""
    elast_calc = ElasticityCalc(
        matpes_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=True,
        relax_deformed_structures=relax_deformed_structures,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )
    # Test Li2O with equilibrium structure
    results = elast_calc.calc(Li2O)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["structure"].lattice.a == pytest.approx(3.291071792359756, rel=1e-1)

    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.4616500809788702, rel=1e-1)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.5064644749775054, rel=1e-1)
    assert results["shear_modulus_vrh"] == pytest.approx(0.40219758881584616, rel=1e-1)
    # Youngs modulus is now self-consistent with bulk/shear (eV/A^3 by default).
    # Previous value used pymatgen's ElasticTensor.y_mod which hardcodes a 9e9
    # GPa->Pa factor that produced incorrect units; see Issue #85.
    assert results["youngs_modulus"] == pytest.approx(0.9338539283876991, rel=1e-1)
    assert results["residuals_sum"] == pytest.approx(3.581519020751326e-08, abs=1e-8)
    assert results["_units"]["bulk_modulus_vrh"] == "eV/A^3"
    assert results["_units"]["youngs_modulus"] == "eV/A^3"

    # Test Li2O without the equilibrium structure
    elast_calc = ElasticityCalc(
        matpes_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=False,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == pytest.approx(2.285e-08, abs=1e-8)

    # Test Li2O with float
    elast_calc = ElasticityCalc(
        matpes_calculator,
        fmax=0.1,
        norm_strains=0.004,
        shear_strains=0.004,
        use_equilibrium=True,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Li2O)
    assert results["residuals_sum"] == 0.0
    assert results["bulk_modulus_vrh"] == pytest.approx(0.4982440620197328, rel=1e-1)


def test_elastic_calc_atoms(
    Si_atoms: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    # Test atoms input. This is not meant to be accurate.
    elast_calc = ElasticityCalc(
        matpes_calculator,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=False,
        relax_structure=False,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    results = elast_calc.calc(Si_atoms)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.5798804241502018, rel=1e-1)


def test_elastic_calc_invalid_states(matpes_calculator: PESCalculator) -> None:
    with pytest.raises(ValueError, match="shear_strains is empty"):
        ElasticityCalc(matpes_calculator, shear_strains=[])
    with pytest.raises(ValueError, match="norm_strains is empty"):
        ElasticityCalc(matpes_calculator, norm_strains=[])

    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(matpes_calculator, norm_strains=[0.0, 0.1])
    with pytest.raises(ValueError, match="strains must be non-zero"):
        ElasticityCalc(matpes_calculator, shear_strains=[0.0, 0.1])


def test_check_and_prelax_raises_on_non_convergence(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    # Force the pre-relax to fail by capping it at a single step with a
    # tight fmax. _check_and_prelax must abort rather than feed an unrelaxed
    # structure into the elasticity calc.
    elast_calc = ElasticityCalc(
        matpes_calculator,
        fmax=1e-8,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        relax_calc_kwargs={"max_steps": 1, "cell_filter": ExpCellFilter},
    )
    with pytest.raises(RuntimeError, match="Pre-relaxation did not converge"):
        elast_calc.calc(Li2O)
