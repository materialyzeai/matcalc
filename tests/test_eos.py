"""Tests for EOSCalc class"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ase.filters import ExpCellFilter

from matcalc import EOSCalc

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_eos_calc(
    Li2O: Structure,
    LiFePO4: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_calc_kwargs={"cell_filter": ExpCellFilter})
    result = eos_calc.calc(Li2O)

    assert result["bulk_modulus_bm"] == pytest.approx(84.08164565338282, rel=1e-1)
    assert result["_units"]["bulk_modulus_bm"] == "GPa"
    assert result["_units"]["eos.volumes"] == "A^3"
    assert result["_units"]["eos.energies"] == "eV"
    assert result["_units"]["energy"] == "eV"
    assert {*result["eos"]} == {"volumes", "energies"}
    assert result["eos"]["volumes"] == pytest.approx(
        [
            18.370519873441342,
            19.622638380261037,
            20.93039763863155,
            22.295007230108368,
            23.71767673624691,
            25.199615738602667,
            26.742033818731066,
            28.34614055818755,
            30.01314553852758,
            31.74425834130665,
            33.54068854808016,
        ],
        rel=1e-1,
    )
    assert result["eos"]["energies"] == pytest.approx(
        [
            -13.49211311340332,
            -13.829581260681152,
            -14.058572769165039,
            -14.203819274902344,
            -14.280638694763184,
            -14.301274299621582,
            -14.275708198547363,
            -14.212100982666016,
            -14.11718463897705,
            -13.99584674835205,
            -13.851357460021973,
        ],
        rel=1e-2,
    )
    eos_calc = EOSCalc(matpes_calculator, relax_structure=False)
    results = list(eos_calc.calc_many([Li2O, LiFePO4]))
    assert len(results) == 2
    assert results[1]["bulk_modulus_bm"] == pytest.approx(69.10044665311513, rel=1e-1)


def test_eos_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for EOSCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.
    eos_calc = EOSCalc(matpes_calculator, fmax=0.1, relax_structure=False)
    result = eos_calc.calc(Si_atoms)

    assert result["bulk_modulus_bm"] == pytest.approx(76.57813407426684, rel=1e-1)
