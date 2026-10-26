"""Tests for the InterfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core import Structure

from matcalc import InterfaceCalc
from matcalc.utils import to_ase_atoms

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core.interface import Interface


@pytest.fixture(scope="module")
def interface(Si: Structure, SiO2: Structure) -> Interface:
    """The first (35-atom, 83 A^2) SiO2(100)/Si(111) interface, with a vacuum over the film."""
    cib = CoherentInterfaceBuilder(
        film_structure=SiO2,
        substrate_structure=Si,
        film_miller=(1, 0, 0),
        substrate_miller=(1, 1, 1),
        zslgen=ZSLGenerator(max_area=100),
    )
    return next(cib.get_interfaces(termination=cib.terminations[0]))


def _area(structure: Structure) -> float:
    matrix = structure.lattice.matrix
    return float(np.linalg.norm(np.cross(matrix[0], matrix[1])))


def _check_energy_identities(res: dict[str, Any], interface: Structure) -> None:
    """interfacial_energy = film + substrate surface energies - adhesion energy, and its explicit formula."""
    area = _area(interface)
    e_interface = res["interface_energy_per_atom"] * len(interface)
    n_film, n_substrate = len(res["final_film_slab"]), len(res["final_substrate_slab"])
    assert n_film + n_substrate == len(interface)
    adhesion = (res["film_slab_energy"] + res["substrate_slab_energy"] - e_interface) / area
    if "adhesion_energy" in res:
        assert res["adhesion_energy"] == pytest.approx(adhesion)
    sigma_film = (res["film_slab_energy"] - n_film * res["film_energy_per_atom"]) / (2 * area)
    sigma_substrate = (res["substrate_slab_energy"] - n_substrate * res["substrate_energy_per_atom"]) / (2 * area)
    assert res["film_surface_energy"] == pytest.approx(sigma_film)
    assert res["substrate_surface_energy"] == pytest.approx(sigma_substrate)
    assert res["interfacial_energy"] == pytest.approx(sigma_film + sigma_substrate - adhesion)
    bulk_energy = n_film * res["film_energy_per_atom"] + n_substrate * res["substrate_energy_per_atom"]
    assert res["interfacial_energy"] == pytest.approx((e_interface - bulk_energy) / area - sigma_film - sigma_substrate)


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
            # Keep only the smallest (83 A^2) match; larger ones add ~25 relaxations of up to 162 atoms.
            zslgen=ZSLGenerator(max_area=100),
        )
    )
    interface_res = results[0]
    assert "final_film" in interface_res
    assert "final_substrate" in interface_res
    assert "final_interface" in interface_res
    assert "adhesion_energy" not in interface_res
    assert interface_res["_units"]["interfacial_energy"] == "eV/A^2"

    assert interface_res["film_energy_per_atom"] == pytest.approx(-7.881573994954427, rel=1e-1)
    assert interface_res["substrate_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert interface_res["film_surface_energy"] == pytest.approx(0.15393912554142708, rel=1e-1)
    assert interface_res["substrate_surface_energy"] == pytest.approx(0.07353920917033396, rel=1e-1)
    assert interface_res["interfacial_energy"] == pytest.approx(0.14388509339553188, rel=1e-1)
    _check_energy_identities(interface_res, interface_res["interface"])


def test_interface_calc_requires_dict(
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` on a non-dict input raises TypeError."""
    interface_calc = InterfaceCalc(calculator=matpes_calculator)
    with pytest.raises(TypeError):
        interface_calc.calc("not a dict")  # type: ignore[arg-type]


def test_interface_calc_from_bulk(
    Si: Structure,
    SiO2: Structure,
    interface: Interface,
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` with bulk inputs (no precomputed energies) and no relaxation reproduces the energies computed from
    independent single points of the bulks, the interface and the film and substrate parts of the interface."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        adhesion_energy=True,
        relax_bulk=False,
        relax_interface=False,
        relax_slab=False,
    )
    res = interface_calc.calc({"interface": interface, "film_bulk": SiO2, "substrate_bulk": Si})

    def energy(structure: Structure) -> float:
        atoms = to_ase_atoms(structure)
        atoms.calc = matpes_calculator
        return float(atoms.get_potential_energy())

    area = _area(interface)
    e_film, e_substrate = energy(interface.film), energy(interface.substrate)
    e_interface = energy(Structure(interface.lattice, interface.species, interface.frac_coords))
    sigma_film = (e_film - len(interface.film) * energy(SiO2) / len(SiO2)) / (2 * area)
    sigma_substrate = (e_substrate - len(interface.substrate) * energy(Si) / len(Si)) / (2 * area)
    adhesion = (e_film + e_substrate - e_interface) / area

    assert res["film_slab_energy"] == pytest.approx(e_film, rel=1e-5)
    assert res["film_surface_energy"] == pytest.approx(sigma_film, rel=1e-4)
    assert res["substrate_surface_energy"] == pytest.approx(sigma_substrate, rel=1e-4)
    assert res["adhesion_energy"] == pytest.approx(adhesion, rel=1e-4)
    assert res["interfacial_energy"] == pytest.approx(sigma_film + sigma_substrate - adhesion, rel=1e-4)
    _check_energy_identities(res, interface)


def test_interface_calc_adhesion(
    Si: Structure, SiO2: Structure, interface: Interface, matpes_calculator: PESCalculator
) -> None:
    """
    Test the adhesion energy of one SiO2/Si interface:
      1) rigid slabs cut from the relaxed interface, together with the interfacial energy
      2) the same slabs relaxed at fixed cell
      3) single points on the as-built interface and slabs given as plain structures
    """
    interface_calc = InterfaceCalc(calculator=matpes_calculator, adhesion_energy=True, relax_slab=False, max_steps=100)
    rigid = interface_calc.calc({"interface": interface, "film_bulk": SiO2, "substrate_bulk": Si})
    assert rigid["interfacial_energy"] == pytest.approx(0.12681154233485015, rel=1e-1)
    assert rigid["adhesion_energy"] == pytest.approx(0.11768947155971884, rel=1e-1)
    assert rigid["final_film_slab"].cart_coords == pytest.approx(
        rigid["final_interface"].cart_coords[interface.film_indices]
    )
    _check_energy_identities(rigid, interface)

    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        interfacial_energy=False,
        adhesion_energy=True,
        relax_interface=False,
        max_steps=100,
    )
    relaxed = interface_calc.calc(
        {
            "interface": rigid["final_interface"],
            "film_slab": rigid["final_film_slab"],
            "substrate_slab": rigid["final_substrate_slab"],
        }
    )
    assert relaxed["adhesion_energy"] == pytest.approx(0.08359310413363714, rel=1e-1)
    assert "interfacial_energy" not in relaxed

    interface_calc = InterfaceCalc(
        calculator=matpes_calculator,
        interfacial_energy=False,
        adhesion_energy=True,
        relax_interface=False,
        relax_slab=False,
    )
    frozen = interface_calc.calc(
        {
            "interface": Structure(interface.lattice, interface.species, interface.frac_coords),
            "film_slab": interface.film,
            "substrate_slab": interface.substrate,
        }
    )
    assert frozen["adhesion_energy"] == pytest.approx(0.04016843476908885, rel=1e-1)


@pytest.mark.parametrize(("given", "missing"), [("film_slab", "substrate_slab"), ("substrate_slab", "film_slab")])
def test_interface_calc_requires_both_slabs(
    Si: Structure, matpes_calculator: PESCalculator, given: str, missing: str
) -> None:
    """Supplying only one of film_slab / substrate_slab raises a ValueError naming the other."""
    interface_calc = InterfaceCalc(calculator=matpes_calculator, adhesion_energy=True)
    with pytest.raises(ValueError, match=f"'{missing}' is missing"):
        interface_calc.calc({"interface": Si, "film_bulk": Si, "substrate_bulk": Si, given: Si})


def test_interface_calc_relax_calc_kwargs_override(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """relax_calc_kwargs may override the arguments InterfaceCalc sets on RelaxCalc (no duplicate-keyword error)."""
    strained = Si.copy()
    strained.scale_lattice(Si.volume * 1.05)
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator, relax_bulk=True, max_steps=5, relax_calc_kwargs={"relax_cell": False}
    )
    out = interface_calc.calc_bulk(strained)
    assert out["final_bulk"].volume == pytest.approx(strained.volume)


@pytest.mark.parametrize(
    ("interfacial_energy", "adhesion_energy"), [(True, False), (False, True)], ids=["interfacial", "adhesion"]
)
def test_interface_calc_requires_labels(
    Si: Structure, matpes_calculator: PESCalculator, interfacial_energy: bool, adhesion_energy: bool
) -> None:
    """Without slabs, a plain Structure (no interface_label site property) raises a ValueError up front for
    either energy, since both need the film and substrate slabs."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator, interfacial_energy=interfacial_energy, adhesion_energy=adhesion_energy
    )
    with pytest.raises(ValueError, match="pass 'film_slab' and 'substrate_slab'"):
        interface_calc.calc({"interface": Si, "film_bulk": Si, "substrate_bulk": Si})


def test_interface_calc_interface_only(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """With both energies off, a plain Structure is accepted and only the interface is evaluated."""
    interface_calc = InterfaceCalc(
        calculator=matpes_calculator, interfacial_energy=False, adhesion_energy=False, relax_interface=False
    )
    res = interface_calc.calc({"interface": Si})
    assert res["num_atoms"] == len(Si)
    assert not {"interfacial_energy", "adhesion_energy", "film_slab_energy"} & res.keys()
