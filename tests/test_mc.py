"""Tests for MCCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from matcalc import MCCalc
from matcalc._intercalation import _RemoveKSites

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

RELAX_KWARGS = {"relax_cell": False}
SINGLE_POINT_ENERGY = 3.0007899233778073


def remove_three(structure: Structure, seed: int = 42) -> _RemoveKSites:
    """Transformation removing 3 random Cu sites without replacement (fixed composition)."""
    return _RemoveKSites(structure.indices_from_symbol("Cu"), 3, np.random.default_rng(seed))


def test_mc_single_point(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """Single-point canonical sampling removes 3 Cu and returns the standard result schema."""
    mc = MCCalc(
        emt_calculator,
        remove_three(Cu_supercell),
        transform_initial=True,
        nsteps=5,
        temperature=1000,
        save_freq=1,
        relax=False,
        seed=42,
        trajfile="mc.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    results = mc.calc(Cu_supercell)

    assert set(results) >= {"final_structure", "energy", "forces", "stress", "acceptance_ratio"}
    assert results["final_structure"].composition.formula == "Cu29"
    assert results["energy"] == pytest.approx(SINGLE_POINT_ENERGY, rel=1e-6)
    assert results["acceptance_ratio"] == pytest.approx(0.4)
    assert results["min_energy"] <= results["energy"]
    assert len(mc.trajectory) == 6  # 1 initial frame + nsteps frames at save_freq=1


def test_mc_relaxation(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """The single relaxation test; all other Monte Carlo tests use single point for speed."""
    mc = MCCalc(
        emt_calculator,
        remove_three(Cu_supercell),
        transform_initial=True,
        nsteps=2,
        temperature=1000,
        save_freq=1,
        relax=True,
        seed=42,
        trajfile="mc_relax.traj",
        relax_calc_kwargs={"relax_cell": False, "max_steps": 20, "fmax": 0.1},
    )
    results = mc.calc(Cu_supercell)

    assert results["final_structure"].composition.formula == "Cu29"
    assert results["energy"] == pytest.approx(2.9606656232203203, rel=1e-5)
    # Relaxation lowers the energy relative to the single-point of the same seeded configuration.
    assert results["energy"] < SINGLE_POINT_ENERGY


def test_mc_fixed_composition(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """transform_initial=True keeps every sampled configuration at the same composition."""
    mc = MCCalc(
        emt_calculator,
        remove_three(Cu_supercell),
        transform_initial=True,
        nsteps=5,
        temperature=1000,
        save_freq=1,
        relax=False,
        seed=7,
        trajfile="mc_fixed.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    mc.calc(Cu_supercell)

    assert {atoms.get_chemical_formula() for atoms in mc.trajectory} == {"Cu29"}


def test_mc_reproducible_seed(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """Identical seeds give identical results; a different seed diverges."""

    def run(seed: int) -> dict:
        return MCCalc(
            emt_calculator,
            remove_three(Cu_supercell, seed=seed),
            transform_initial=True,
            nsteps=5,
            temperature=1000,
            save_freq=5,
            relax=False,
            seed=seed,
            trajfile="mc_seed.traj",
            relax_calc_kwargs=RELAX_KWARGS,
        ).calc(Cu_supercell)

    first, repeat, other = run(42), run(42), run(123)
    assert first["energy"] == pytest.approx(repeat["energy"])
    assert first["acceptance_ratio"] == repeat["acceptance_ratio"]
    assert first["energy"] != pytest.approx(other["energy"])


def test_mc_grand_canonical_not_implemented(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """The reserved grand-canonical ensemble raises NotImplementedError."""
    mc = MCCalc(
        emt_calculator,
        remove_three(Cu_supercell),
        transform_initial=True,
        ensemble="grand_canonical",
        nsteps=1,
        relax=False,
        seed=42,
        trajfile="mc_gc.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    with pytest.raises(NotImplementedError):
        mc.calc(Cu_supercell)


def test_mc_invalid_ensemble(Cu_supercell: Structure, emt_calculator: Calculator) -> None:
    """An unrecognized ensemble raises ValueError."""
    mc = MCCalc(
        emt_calculator,
        remove_three(Cu_supercell),
        transform_initial=True,
        ensemble="bogus",  # type: ignore[arg-type]
        nsteps=1,
        relax=False,
        seed=42,
        trajfile="mc_bad.traj",
        relax_calc_kwargs=RELAX_KWARGS,
    )
    with pytest.raises(ValueError, match="canonical"):
        mc.calc(Cu_supercell)
