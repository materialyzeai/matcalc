"""Tests for the InterfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import InterfaceCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_interface_calc_basic(Si: Structure, SiO2: Structure, matpes_calculator: PESCalculator) -> None:
    """
    Test the basic workflow:
      1) calc_interfaces on SiO2 (film) and Si (substrate)
      2) calc on the resulting interfaces
      3) Check the final results
    """
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_interface=True,
        fmax=0.1,
        max_steps=100,
    )

    results = list(
        interface_calc.calc_interfaces(
            film_bulk=SiO2,
            substrate_bulk=Si,
            film_miller=(1, 0, 0),
            substrate_miller=(1, 1, 1),
        )
    )
    interface_res = results[0]
    assert "final_film" in interface_res
    assert "final_substrate" in interface_res
    assert "final_interface" in interface_res

    assert interface_res["film_energy_per_atom"] == pytest.approx(-7.881573994954427, rel=1e-1)
    assert interface_res["substrate_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert interface_res["interfacial_energy"] == pytest.approx(0.15930456535294427, rel=1e-1)
