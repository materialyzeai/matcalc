from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc._stability import EnergeticsCalc

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_energetics_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    result = EnergeticsCalc(matpes_calculator).calc(Li2O)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    assert result["formation_energy_per_atom"] == pytest.approx(-1.7850605646769206, rel=1e-2)
    assert result["cohesive_energy_per_atom"] == pytest.approx(-4.053125177202149, rel=1e-2)
    assert result["_units"]["formation_energy_per_atom"] == "eV/atom"
    assert result["_units"]["cohesive_energy_per_atom"] == "eV/atom"
    assert result["_units"]["energy"] == "eV"

    # Note that the value differs from MP primarily because of the correction. A correction of -0.70 eV is applied to
    # each O atom, which accounts almost entirely for the difference between this predicted formation energy and the
    # MP calculated value of -2.0 eV.

    result = EnergeticsCalc(matpes_calculator, use_gs_reference=True).calc(Li2O)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    assert result["formation_energy_per_atom"] == pytest.approx(-1.8454608797021486, rel=1e-2)
    assert result["cohesive_energy_per_atom"] == pytest.approx(-4.053125177202149, rel=1e-2)


def test_energetics_calc_element_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    result = EnergeticsCalc(matpes_calculator).calc(Si_atoms)
    for key in (
        "final_structure",
        "formation_energy_per_atom",
        "cohesive_energy_per_atom",
    ):
        assert key in result, f"{key=} not in result"
    # This is an element. The formation energy should be close to 0.
    assert result["formation_energy_per_atom"] == pytest.approx(0, abs=1e-2)
