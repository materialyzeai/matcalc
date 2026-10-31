"""Tests for IntercalationCalc class."""

from __future__ import annotations

import warnings
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
    )
    results = calc.calc(Cu.copy())

    assert list(results) == ["0"]
    res = results["0"]
    assert res["Num_removed"] == 3
    assert res["concentration"] == pytest.approx(3 / 32)
    assert res["final_structure"].composition.formula == "Cu29"
    assert res["energy"] == pytest.approx(3.1541968844142314, rel=1e-6)
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
    )
    res = calc.calc(Cu.copy())["0"]

    assert res["Num_removed"] == 3
    assert res["concentration"] == pytest.approx(3 / 32)
    assert res["final_structure"].composition.formula == "Cu29"
    assert res["energy"] == pytest.approx(3.209710325848368, rel=1e-6)
    assert res["min_energy"] <= res["energy"]

    frames = read("ic_markov_k3.traj", ":")
    assert {atoms.get_chemical_formula() for atoms in frames} == {"Cu29"}

    def occupancy(atoms: Atoms) -> set:
        return {tuple(coord) for coord in np.round(atoms.get_scaled_positions(), 3).tolist()}

    moves = [len(occupancy(frames[i]) ^ occupancy(frames[i + 1])) for i in range(len(frames) - 1)]
    assert max(moves) <= 2  # each accepted step relocates at most one Cu (a local swap)


def test_intercalation_relax_kwargs_ignored_without_relax(Cu: Structure, emt_calculator: Calculator) -> None:
    """relax_calc_kwargs with relax=False warns once at construction, not again for every concentration level."""
    with pytest.warns(UserWarning, match="ignored when relax=False"):
        calc = IntercalationCalc(
            emt_calculator,
            nsteps=2,
            temperature=1000,
            concentration_range=[0.05, 0.10, 0.03],  # -> k = 2, 3
            species="Cu",
            relax=False,
            seed=42,
            supercell=[2, 2, 2],
            trajfile="ic_kwargs.traj",
            relax_calc_kwargs={"max_steps": 50},
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="relax_calc_kwargs is ignored")
        results = calc.calc(Cu.copy())
    assert len(results) == 2


def test_intercalation_endpoints_skip_mc(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """Levels with a single configuration (k = 0 and k = all candidates) are scored once, without MC steps."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=3,
        temperature=1000,
        concentration_range=[0.0, 1.01, 1 / 3],  # -> k = 0, 1, 2, 3
        indices=[0, 1, 2],
        relax=False,
        seed=42,
        save_freq=1,
        trajfile="ic_ends.traj",
    )
    results = calc.calc(Cu_supercell.copy())

    assert [res["Num_removed"] for res in results.values()] == [0, 1, 2, 3]
    for key, n_frames in (("0", 1), ("1", 4), ("2", 4), ("3", 1)):
        k = results[key]["Num_removed"]
        assert len(read(f"ic_ends_k{k}.traj", index=":")) == n_frames
    assert np.isnan(results["0"]["acceptance_ratio"])
    assert np.isnan(results["3"]["acceptance_ratio"])
    assert results["0"]["final_structure"].composition.formula == "Cu32"
    assert results["3"]["final_structure"].composition.formula == "Cu29"


def test_intercalation_markov_fully_occupied(Cu: Structure, emt_calculator: Calculator) -> None:
    """A Markov level with nothing removed has no vacancy to swap into, so it is scored without MC steps."""
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=3,
        temperature=1000,
        concentration_range=[0.0, 0.01, 0.1],  # -> k = 0
        species="Cu",
        supercell=[2, 2, 2],
        algorithm="markov",
        relax=False,
        seed=42,
        trajfile="ic_markov_full.traj",
    )
    res = calc.calc(Cu.copy())["0"]
    assert res["final_structure"].composition.formula == "Cu32"
    assert np.isnan(res["acceptance_ratio"])


def test_intercalation_input_unchanged_and_reusable(Cu: Structure, emt_calculator: Calculator) -> None:
    """calc() leaves the caller's structure untouched, so the same calculator can run twice with supercell."""
    host = Cu.copy()
    calc = IntercalationCalc(
        emt_calculator,
        nsteps=1,
        temperature=1000,
        concentration_range=[0.09, 0.10, 0.05],  # -> k = 3 of 32
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic_reuse.traj",
    )
    first = calc.calc(host)
    assert len(host) == len(Cu)
    second = calc.calc(host)
    assert len(host) == len(Cu)
    for res in (first["0"], second["0"]):
        assert res["Num_removed"] == 3
        assert res["final_structure"].composition.formula == "Cu29"


def test_intercalation_explicit_concentrations(Cu: Structure, emt_calculator: Calculator) -> None:
    """Explicit concentrations give the same levels and energies as the equivalent concentration_range."""
    common = {
        "nsteps": 2,
        "temperature": 1000,
        "species": "Cu",
        "relax": False,
        "seed": 42,
        "supercell": [2, 2, 2],
        "write_results": False,
    }
    from_range = IntercalationCalc(
        emt_calculator, concentration_range=[0.05, 0.10, 0.03], trajfile="ic_range.traj", **common
    ).calc(Cu.copy())
    explicit = IntercalationCalc(
        emt_calculator, concentrations=[2 / 32, 3 / 32], trajfile="ic_explicit.traj", **common
    ).calc(Cu.copy())

    assert [r["Num_removed"] for r in explicit.values()] == [2, 3]
    for key in from_range:
        assert explicit[key]["energy"] == pytest.approx(from_range[key]["energy"])


@pytest.mark.parametrize("kwargs", [{}, {"concentration_range": [0.1, 0.2, 0.1], "concentrations": [0.1]}])
def test_intercalation_concentrations_exclusive(emt_calculator: Calculator, kwargs: dict) -> None:
    """Exactly one of concentration_range and concentrations must be given."""
    with pytest.raises(ValueError, match="exactly one of concentration_range and concentrations"):
        IntercalationCalc(emt_calculator, nsteps=1, temperature=300, species="Cu", relax=False, **kwargs)


@pytest.mark.parametrize(("write_results", "path"), [(False, None), ("levels.json.gz", "levels.json.gz")])
def test_intercalation_write_results(
    Cu: Structure, emt_calculator: Calculator, write_results: bool | str, path: str | None
) -> None:
    """write_results=False skips the results file; a string selects its path."""
    IntercalationCalc(
        emt_calculator,
        nsteps=1,
        temperature=1000,
        concentrations=[3 / 32],
        species="Cu",
        relax=False,
        seed=42,
        supercell=[2, 2, 2],
        trajfile="ic_write.traj",
        write_results=write_results,
    ).calc(Cu.copy())

    assert not Path("results.json.gz").exists()
    if path is not None:
        assert loadfn(path)["0"]["Num_removed"] == 3
