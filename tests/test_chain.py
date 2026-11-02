"""Tests for ElasticCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ase.filters import ExpCellFilter

from matcalc import ChainedCalc, ElasticityCalc, EnergeticsCalc, PropCalc, RelaxCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
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


class _Tag(PropCalc):
    """Adds a key to the result; raises for inputs whose structure is listed in ``fail_on``."""

    def __init__(self, key: str, fail_on: tuple = ()) -> None:
        self.key = key
        self.fail_on = fail_on

    def calc(self, structure: Any) -> dict[str, Any]:
        result = super().calc(structure)
        if result["final_structure"] in self.fail_on:
            raise ValueError(f"failed on {result['final_structure']}")
        return result | {self.key: True}


def test_chain_calc_many_allow_errors() -> None:
    """A structure that fails at an early step yields None, not a partial dict from the later steps."""
    chain = ChainedCalc([_Tag("first", fail_on=("bad",)), _Tag("second")])
    results = list(chain.calc_many([{"structure": "good"}, {"structure": "bad"}], allow_errors=True))
    assert results[0] == {"structure": "good", "final_structure": "good", "first": True, "second": True}
    assert results[1] is None

    with pytest.raises(ValueError, match="failed on bad"):
        list(chain.calc_many([{"structure": "bad"}]))


def test_chain_calc_many_parallel(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """Multi-step chains used to deadlock in calc_many with n_jobs > 1."""
    chain = ChainedCalc([RelaxCalc(matpes_calculator, relax_atoms=False, relax_cell=False) for _ in range(3)])
    structures = [Si] * 32
    serial = list(chain.calc_many(structures))
    parallel = list(chain.calc_many(structures, n_jobs=2))
    assert len(parallel) == len(structures)
    for res_serial, res_parallel in zip(serial, parallel, strict=True):
        assert res_parallel["energy"] == pytest.approx(res_serial["energy"], abs=1e-5)
