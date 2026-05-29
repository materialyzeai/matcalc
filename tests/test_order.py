"""Tests for OrderCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymatgen.core import Lattice, Structure

from matcalc import OrderCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator


@pytest.fixture(scope="module")
def disordered_CuAu() -> Structure:
    """Disordered Cu0.5Au0.5 fcc supercell (8 sites)."""
    base = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.7), [{"Cu": 0.5, "Au": 0.5}], [[0, 0, 0]])
    return base * (2, 1, 1)


def test_order_calc(disordered_CuAu: Structure, matpes_calculator: PESCalculator) -> None:
    """Random ordering + Monte Carlo produces a fully ordered, composition-conserving structure."""
    order_calc = OrderCalc(matpes_calculator, nsteps=10, temperature=1000.0, seed=42)
    result = order_calc.calc(disordered_CuAu)

    final = result["final_structure"]
    assert final.is_ordered
    # Composition is conserved through swaps.
    assert final.composition.reduced_formula == disordered_CuAu.composition.reduced_formula
    assert len(final) == len(disordered_CuAu)

    assert isinstance(result["energy"], float)
    assert len(result["energies"]) == 11  # nsteps + 1
    assert result["energy"] == min(result["energies"])  # final_structure is the best ordering found
    assert 0.0 <= result["acceptance_ratio"] <= 1.0
    assert result["_units"]["energy"] == "eV"


def test_order_calc_deterministic(disordered_CuAu: Structure, matpes_calculator: PESCalculator) -> None:
    """Same seed gives identical results."""
    res1 = OrderCalc(matpes_calculator, nsteps=8, seed=7).calc(disordered_CuAu)
    res2 = OrderCalc(matpes_calculator, nsteps=8, seed=7).calc(disordered_CuAu)
    assert res1["energies"] == res2["energies"]


def test_order_calc_vacancies(matpes_calculator: PESCalculator) -> None:
    """Partial occupancy below 1 is ordered with vacancies (sites removed)."""
    disordered = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.2), [{"Cu": 0.75}], [[0, 0, 0]])
    result = OrderCalc(matpes_calculator, nsteps=5, seed=1).calc(disordered)
    final = result["final_structure"]
    assert final.is_ordered
    assert len(final) == 3  # 4 fcc sites, 25% vacancies -> 3 Cu


def test_order_calc_already_ordered(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """An already-ordered structure raises."""
    with pytest.raises(ValueError, match="already ordered"):
        OrderCalc(matpes_calculator, nsteps=5).calc(Si)


def test_order_calc_negative_temperature(matpes_calculator: PESCalculator) -> None:
    """Negative temperature is rejected at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        OrderCalc(matpes_calculator, temperature=-1.0)
