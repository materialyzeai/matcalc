"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.filters import ExpCellFilter

from matcalc import ChainedCalc, ElasticityCalc, EnergeticsCalc, RelaxCalc

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_chain_calc(
    Li2O: Structure,
) -> None:
    """Tests for ElasticCalc class"""
    potential = "TensorNet-PES-MatPES-PBE-2025.2"

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

    for expected_keys in (
        "elastic_tensor",
        "structure",
        "bulk_modulus_vrh",
        "shear_modulus_vrh",
        "youngs_modulus",
        "residuals_sum",
    ):
        assert expected_keys in results
        assert results[expected_keys] is not None

    results = list(calc.calc_many([Li2O] * 2))
    assert len(results) == 2
