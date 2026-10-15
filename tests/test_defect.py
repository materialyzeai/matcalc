"""Tests for neutral point-defect formation energies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from monty.serialization import loadfn

from matcalc import DefectCalc, RelaxCalc
from matcalc._stability import ELEMENTAL_REFS_DIR

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from matcalc.utils import PESCalculator


def test_calc_defects_vacancy(Cu: Structure, matpes_calculator: PESCalculator) -> None:
    [result] = DefectCalc(matpes_calculator, fmax=0.05).calc_defects(Cu, supercell_matrix=2)

    # Independent protocol: relax the conventional cell, then relax the vacancy supercell at fixed cell.
    bulk = RelaxCalc(matpes_calculator, fmax=0.05, optimizer="BFGS").calc(Cu)
    vacancy = bulk["final_structure"].make_supercell(2, in_place=False)
    vacancy.remove_sites([0])
    defect = RelaxCalc(matpes_calculator, fmax=0.05, optimizer="BFGS", relax_cell=False).calc(vacancy)
    e_bulk = bulk["energy"] / 4

    assert result["defect_name"] == "v_Cu"
    assert result["element_changes"] == {"Cu": -1}
    assert len(result["defect"]) == 31
    np.testing.assert_array_equal(result["supercell_matrix"], 2 * np.eye(3))
    np.testing.assert_allclose(np.round(result["defect_frac_coords"], 6) % 1, [0, 0, 0], atol=1e-6)
    assert result["bulk_energy_per_atom"] == pytest.approx(e_bulk)
    assert result["chemical_potentials"] == {"Cu": pytest.approx(e_bulk)}
    assert result["defect_energy"] == pytest.approx(defect["energy"], abs=1e-4)
    assert result["formation_energy"] == pytest.approx(defect["energy"] - 31 * e_bulk, abs=1e-4)
    assert 0.5 < result["formation_energy"] < 2
    np.testing.assert_allclose(result["final_defect"].lattice.matrix, vacancy.lattice.matrix, atol=1e-6)
    assert not np.allclose(result["final_defect"].cart_coords, vacancy.cart_coords)
    assert result["_units"]["formation_energy"] == "eV"
    assert result["_units"]["bulk_energy_per_atom"] == "eV/atom"


def test_defect_calc_manual(Cu: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DefectCalc(matpes_calculator, fmax=0.05)
    [generated] = calc.calc_defects(Cu, supercell_matrix=2)
    e_bulk = generated["bulk_energy_per_atom"]
    divacancy = generated["final_bulk"].make_supercell(2, in_place=False)
    divacancy.remove_sites([0, 1])

    from_bulk = calc.calc({"defect": divacancy, "bulk": Cu, "element_changes": {"Cu": -2}, "tag": "di"})
    from_energy = calc.calc(
        {
            "defect": divacancy,
            "element_changes": {"Cu": -2},
            "final_bulk": generated["final_bulk"],
            "bulk_energy_per_atom": e_bulk,
        }
    )
    unrelaxed = DefectCalc(matpes_calculator, relax_defect=False).calc(
        {"defect": divacancy, "element_changes": {"Cu": -2}, "final_bulk": Cu, "bulk_energy_per_atom": e_bulk}
    )

    assert from_bulk["tag"] == "di"
    assert from_bulk["bulk_energy_per_atom"] == pytest.approx(e_bulk)
    assert from_bulk["formation_energy"] == pytest.approx(from_energy["formation_energy"], abs=1e-4)
    assert from_energy["formation_energy"] == pytest.approx(from_energy["defect_energy"] - 30 * e_bulk)
    assert from_energy["formation_energy"] > 0
    np.testing.assert_allclose(unrelaxed["final_defect"].cart_coords, divacancy.cart_coords)
    assert unrelaxed["defect_energy"] == pytest.approx(matpes_calculator.get_potential_energy(divacancy.to_ase_atoms()))
    assert unrelaxed["formation_energy"] > from_energy["formation_energy"]


def test_calc_defects_compound(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    mu = {"Li": -1.9, "O": -4.9}
    calc = DefectCalc(matpes_calculator, fmax=0.1, max_steps=20, chemical_potentials=mu)
    results = calc.calc_defects(Li2O, supercell_matrix=1, substitutions={"Li": "Na"}, antisites=True)

    # Na comes from the MatPES-PBE elemental references relaxed with the same calculator.
    na_ref = loadfn(ELEMENTAL_REFS_DIR / "MatPES-PBE-Element-Refs.json.gz")["Na"]["structure"]
    na = RelaxCalc(matpes_calculator, fmax=0.1, max_steps=20, optimizer="BFGS").calc(na_ref)
    mu["Na"] = na["energy"] / len(na["final_structure"])

    by_name = {r["defect_name"]: r for r in results}
    assert list(by_name) == ["v_Li", "Na_Li", "O_Li", "v_O", "Li_O"]
    assert by_name["O_Li"]["element_changes"] == {"Li": -1, "O": 1}
    assert by_name["Na_Li"]["chemical_potentials"] == pytest.approx({"Li": mu["Li"], "Na": mu["Na"]})
    e_bulk = results[0]["bulk_energy_per_atom"]
    for r in results:
        changes = r["element_changes"]
        expected_counts = (Li2O.composition.element_composition * 4).get_el_amt_dict()
        for el, n in changes.items():
            expected_counts[el] = expected_counts.get(el, 0) + n
        assert r["defect"].composition.get_el_amt_dict() == {el: n for el, n in expected_counts.items() if n}
        n_pristine = len(r["defect"]) - sum(changes.values())
        expected = r["defect_energy"] - n_pristine * e_bulk - sum(n * mu[el] for el, n in changes.items())
        assert r["formation_energy"] == pytest.approx(expected)

    with pytest.raises(ValueError, match="element_changes"):
        calc.calc({"defect": results[0]["defect"], "element_changes": {"O": -1}, "bulk": Li2O})


def test_calc_defects_interstitials(Cu: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DefectCalc(matpes_calculator, fmax=0.05, chemical_potentials={"H": -1.0})
    results = calc.calc_defects(
        Cu, supercell_matrix=(2, 1, 1), vacancies=False, interstitials={"H": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]}
    )

    assert [r["defect_name"] for r in results] == ["H_i_1", "H_i_2"]
    assert results[0]["element_changes"] == {"H": 1}
    assert [len(r["defect"]) for r in results] == [9, 9]
    np.testing.assert_allclose(results[0]["defect_frac_coords"], [0.25, 0.5, 0.5])
    np.testing.assert_allclose(results[1]["defect_frac_coords"], [0.125, 0.25, 0.25])
    for r in results:
        expected = r["defect_energy"] - 8 * r["bulk_energy_per_atom"] + 1.0
        assert r["formation_energy"] == pytest.approx(expected)


def test_calc_defects_voronoi_interstitials(Cu: Structure, matpes_calculator: PESCalculator) -> None:
    pytest.importorskip("pymatgen.analysis.defects.generators")
    calc = DefectCalc(matpes_calculator, relax_bulk=False, relax_defect=False, chemical_potentials={"H": -1.0})
    results = calc.calc_defects(Cu, supercell_matrix=1, vacancies=False, interstitials="H")

    # Tetrahedral and octahedral sites of fcc Cu.
    distances = sorted(
        min(np.linalg.norm(r["defect"][-1].coords - site.coords) for site in r["defect"][:-1]) for r in results
    )
    assert distances == pytest.approx([3.6 * np.sqrt(3) / 4, 3.6 / 2], abs=0.05)
    assert [r["defect_name"] for r in results] == ["H_i_1", "H_i_2"]


def test_calc_defects_supercell(Si: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DefectCalc(matpes_calculator, relax_bulk=False, relax_defect=False)
    [result] = calc.calc_defects(Si)

    # The 8-atom conventional cell is repeated twice along each axis to reach min_length = 10 A.
    np.testing.assert_array_equal(result["supercell_matrix"], 2 * np.eye(3))
    assert len(result["final_bulk"]) == 8
    assert len(result["defect"]) == 63
    np.testing.assert_allclose(result["final_defect"].cart_coords, result["defect"].cart_coords)
    assert np.isfinite(result["formation_energy"])
    # A single element symbol is one element, not a sequence of characters.
    assert [r["defect_name"] for r in calc.calc_defects(Si, supercell_matrix=1, vacancies="Si")] == ["v_Si"]


def test_defect_calc_invalid_inputs(Si: Structure, Cu: Structure, matpes_calculator: PESCalculator) -> None:
    calc = DefectCalc(matpes_calculator, elemental_refs={})
    with pytest.raises(ValueError, match="must be a dict"):
        calc.calc(Si)
    with pytest.raises(ValueError, match="must be a dict"):
        calc.calc({"defect": Si, "bulk_energy_per_atom": -5.0, "element_changes": {}})
    with pytest.raises(ValueError, match=r"\['Ag'\] are not in the host Cu"):
        calc.calc_defects(Cu, substitutions={"Ag": "Cu"})
    with pytest.raises(ValueError, match="No elemental reference for Ni"):
        calc.calc_defects(Cu, supercell_matrix=1, vacancies=False, substitutions={"Cu": "Ni"})


def test_defect_calc_si(Si: Structure, matpes_calculator: PESCalculator) -> None:
    [result] = DefectCalc(matpes_calculator, fmax=0.05, max_steps=30).calc_defects(Si, supercell_matrix=1)

    assert result["defect_name"] == "v_Si"
    assert len(result["defect"]) == 7
    assert 1 < result["formation_energy"] < 6
