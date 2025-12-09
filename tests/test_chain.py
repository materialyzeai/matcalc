"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.filters import ExpCellFilter

from matcalc import ChainedCalc, ElasticityCalc, EnergeticsCalc, RelaxCalc

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_chain_calc(
    Li2O: Structure,
) -> None:
    """Tests for ElasticCalc class"""
    potential = "TensorNet-MatPES-PBE-v2025.1-PES"

    relax_calc = RelaxCalc(
        potential,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
    )
    energetics_calc = EnergeticsCalc(potential, relax_structure=False)
    elast_calc = ElasticityCalc(
        potential,
        fmax=0.1,
        norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
        shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
        use_equilibrium=True,
        relax_structure=False,
        relax_deformed_structures=True,
        relax_calc_kwargs={"cell_filter": ExpCellFilter},
    )

    calc = ChainedCalc([relax_calc, energetics_calc, elast_calc])
    # Test Li2O with equilibrium structure
    results = calc.calc(Li2O)
    assert results["elastic_tensor"].shape == (3, 3, 3, 3)
    assert results["structure"].lattice.a == pytest.approx(3.291071792359756, rel=1e-1)

    assert results["elastic_tensor"][0][1][1][0] == pytest.approx(0.39659910398768233, rel=1e-1)
    assert results["bulk_modulus_vrh"] == pytest.approx(0.40163873655210464, rel=1e-1)
    assert results["shear_modulus_vrh"] == pytest.approx(0.2695915414932737, rel=1e-1)
    assert results["youngs_modulus"] == pytest.approx(660904125.9537675, rel=1e-1)
    assert results["residuals_sum"] == pytest.approx(1.4241076180128878e-08, abs=1e-6)
    # A chained calculation has results from all steps.

    assert results["energy"] == pytest.approx(-14.176680, rel=1e-1)
    assert results["a"] == pytest.approx(3.291072, rel=1e-1)
    assert results["alpha"] == pytest.approx(60, abs=5)

    assert results["formation_energy_per_atom"] == pytest.approx(-1.8037596543629963, abs=1e-3)

    results = list(calc.calc_many([Li2O] * 2))
    assert len(results) == 2
    assert results[0]["energy"] == pytest.approx(-14.176680, rel=1e-1)
