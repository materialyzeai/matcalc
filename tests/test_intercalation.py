"""Tests for IntercalationCalc class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.io import read
from monty.serialization import loadfn
from pymatgen.core import Structure

from matcalc import IntercalationCalc

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

RELAX_KWARGS = {"relax_cell": False}


def test_intercalation_single_point(Cu: Structure, emt_calculator: Calculator) -> None:
    """A single concentration level removes 3 Cu and returns the per-level result schema."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=5,
        temperature=1000,
        concentration_range=[0.09, 0.10, 0.05],  # -> k = round(0.094 * 32) = 3
        save_freq=1,
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    results = calc.calc(Cu.copy())

    assert list(results) == ["0"]
    res = results["0"]
    assert res["Num_removed"] == 3
    assert res["concentration"] == pytest.approx(3 / 32)
    assert res["final_structure"].composition.formula == "Cu29"
    assert res["energy"] == pytest.approx(3.0007899233778073, rel=1e-6)
    assert 0 <= res["acceptance_ratio"] <= 1
    assert Path("ic_k3.traj").exists()
    assert Path("results.json.gz").exists()


def test_intercalation_concentration_sweep(Cu: Structure, emt_calculator: Calculator) -> None:
    """A concentration range yields one result and trajectory per vacancy count."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=4,
        temperature=1000,
        concentration_range=[0.06, 0.12, 0.03],  # -> ks = [2, 3]
        save_freq=1,
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic_sweep.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    results = calc.calc(Cu.copy())

    assert list(results) == ["0", "1"]
    assert [results[k]["Num_removed"] for k in results] == [2, 3]
    assert results["0"]["final_structure"].composition.formula == "Cu30"
    assert results["1"]["final_structure"].composition.formula == "Cu29"
    assert Path("ic_sweep_k2.traj").exists()
    assert Path("ic_sweep_k3.traj").exists()


def test_intercalation_fixed_composition_per_k(Cu: Structure, emt_calculator: Calculator) -> None:
    """Every frame within a per-k trajectory shares one composition (no vacancy drift)."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=4,
        temperature=1000,
        concentration_range=[0.06, 0.12, 0.03],
        save_freq=1,
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic_fix.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    calc.calc(Cu.copy())

    for k, formula in ((2, "Cu30"), (3, "Cu29")):
        frames = read(f"ic_fix_k{k}.traj", ":")
        assert {atoms.get_chemical_formula() for atoms in frames} == {formula}


def test_intercalation_results_serialized(Cu: Structure, emt_calculator: Calculator) -> None:
    """The per-level results round-trip through results.json.gz with reconstructed structures."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=3,
        temperature=1000,
        concentration_range=[0.09, 0.10, 0.05],
        save_freq=1,
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic_ser.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    calc.calc(Cu.copy())

    loaded = loadfn("results.json.gz")
    assert isinstance(loaded["0"]["final_structure"], Structure)
    assert loaded["0"]["Num_removed"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Provide one of indices or species"),
        ({"indices": [0, 1, 2], "supercell": [2, 2, 2]}, "supercell"),
    ],
)
def test_intercalation_validation(
    Cu: Structure,
    emt_calculator: Calculator,
    kwargs: dict,
    match: str,
) -> None:
    """Invalid species/indices/supercell combinations raise ValueError."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=1,
        temperature=1000,
        concentration_range=[0.1, 0.2, 0.1],
        relax=False,
        seed=42,
        trajfile="ic_val.traj",
        relax_calc_kwargs=RELAX_KWARGS,
        **kwargs,
    )
    with pytest.raises(ValueError, match=match):
        calc.calc(Cu.copy())


def test_intercalation_markov(Cu: Structure, emt_calculator: Calculator) -> None:
    """Markov mode samples a true local chain (one swap per accepted step) at fixed composition."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=6,
        temperature=2000,
        concentration_range=[0.09, 0.10, 0.05],
        save_freq=1,
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        algorithm="markov",
        swap_size=1,
        trajfile="ic_markov.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    res = calc.calc(Cu.copy())["0"]

    assert res["Num_removed"] == 3
    assert res["concentration"] == pytest.approx(3 / 32)
    assert res["final_structure"].composition.formula == "Cu29"
    assert res["energy"] == pytest.approx(3.2843720430568393, rel=1e-6)
    assert res["min_energy"] <= res["energy"]

    frames = read("ic_markov_k3.traj", ":")
    assert {atoms.get_chemical_formula() for atoms in frames} == {"Cu29"}

    def occupancy(atoms: Atoms) -> set:
        return {tuple(coord) for coord in np.round(atoms.get_scaled_positions(), 3).tolist()}

    moves = [len(occupancy(frames[i]) ^ occupancy(frames[i + 1])) for i in range(len(frames) - 1)]
    assert max(moves) <= 2  # each accepted step relocates at most one Cu (a local swap)
