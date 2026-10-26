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
    assert interface_res["interfacial_energy"] == pytest.approx(0.19152603543013672, rel=1e-1)


def test_interface_calc_requires_dict(
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` on a non-dict input raises TypeError."""
    interface_calc = InterfaceCalc(calculator=matpes_calculator)
    with pytest.raises(TypeError):
        interface_calc.calc("not a dict")  # type: ignore[arg-type]


def test_interface_calc_from_bulk_film_specific(
    Si: Structure,
    SiO2: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` with bulk inputs (no precomputed energies) derives counts via the
    film-specific element branch of the composition fallback."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=False,
        relax_interface=False,
        fmax=100.0,
        max_steps=1,
    )
    # Film-specific branch: film_bulk (SiO2) supplies the unique element O that
    # substrate_bulk (Si) lacks, so film_in_interface = 0 and substrate_in_interface
    # holds all atoms. NOTE: the interface itself is pure Si -- it is NOT the film --
    # so this is a degenerate (zero film atoms) input, valid only for branch coverage,
    # not physically meaningful; do not copy it as a usage example.
    structure = {
        "interface": Si.copy(),
        "film_bulk": SiO2,
        "substrate_bulk": Si,
    }
    out = interface_calc.calc(structure)
    # film_in_interface = 0 and, with no relaxation, the interface equals the Si
    # substrate, so the gamma numerator vanishes -> interfacial_energy ~ 0.
    assert out["interfacial_energy"] == pytest.approx(0.0, abs=1e-6)


def test_interface_calc_from_bulk_substrate_specific(
    Si: Structure,
    SiO2: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Composition fallback via the substrate-specific element branch."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=False,
        relax_interface=False,
        fmax=100.0,
        max_steps=1,
    )
    # Substrate-specific branch: substrate_bulk (SiO2) supplies the unique element
    # O that film_bulk (Si) lacks, so substrate_in_interface = 0 and film_in_interface
    # holds all atoms. NOTE: the interface itself is pure Si -- it is NOT the substrate --
    # so this is a degenerate (zero substrate atoms) input, valid only for branch
    # coverage, not physically meaningful; do not copy it as a usage example.
    structure = {
        "interface": Si.copy(),
        "film_bulk": Si,
        "substrate_bulk": SiO2,
    }
    out = interface_calc.calc(structure)
    # substrate_in_interface = 0 and, with no relaxation, the interface equals the Si
    # film, so the gamma numerator vanishes -> interfacial_energy ~ 0.
    assert out["interfacial_energy"] == pytest.approx(0.0, abs=1e-6)


def test_interface_calc_shared_element_set(
    Si: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """When film and substrate share the full elemental set and the interface exposes
    no film/substrate indices, ``calc`` cannot split the atoms and raises ValueError."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=False,
        relax_interface=False,
        fmax=100.0,
        max_steps=1,
    )
    structure = {
        "interface": Si.copy(),
        "film_bulk": Si,
        "substrate_bulk": Si,
    }
    with pytest.raises(ValueError, match="Could not determine"):
        interface_calc.calc(structure)
