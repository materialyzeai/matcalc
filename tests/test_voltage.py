"""Tests for VoltageCalc class."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from matcalc import ChainedCalc, RelaxCalc, VoltageCalc

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from matcalc.utils import PESCalculator

# EMT has no Li, so Cu is the working ion in an L1_2 Cu3Au host (Au corners, Cu face centres).
# With supercell [2, 1, 1]: 8 atoms, 6 Cu sites, so 7 ion counts.
N_SITES = 6
CU_VALENCE = 4  # pymatgen convention: highest oxidation state of Cu


@pytest.fixture(scope="module")
def Cu3Au() -> Structure:
    """L1_2 Cu3Au conventional cell (4 atoms)."""
    return Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.75), ["Au", "Cu"], [[0, 0, 0], [0, 0.5, 0.5]])


def emt_voltage_calc(emt_calculator: Calculator, **kwargs: object) -> VoltageCalc:
    """Fast single-point VoltageCalc on the Cu3Au host."""
    defaults: dict = {
        "species": "Cu",
        "supercell": [2, 1, 1],
        "nsteps": 3,
        "temperature": 1000,
        "seed": 42,
        "relax": False,
        "relax_structure": False,
        "ion_reference": -0.01,
        "trajfile": "v.traj",
    }
    return VoltageCalc(emt_calculator, **(defaults | kwargs))  # type: ignore[arg-type]


def test_voltage_profile(Cu3Au: Structure, emt_calculator: Calculator) -> None:
    """Every ion count is sampled and the voltages follow from the lowest energies."""
    res = emt_voltage_calc(emt_calculator).calc(Cu3Au)

    np.testing.assert_allclose(res["x"], np.arange(N_SITES + 1) / N_SITES)
    np.testing.assert_array_equal(res["n_removed"], np.arange(N_SITES, -1, -1))
    assert np.isnan(res["acceptance_ratios"][[0, -1]]).all()  # single-configuration end points
    assert res["mixing_energies"][[0, -1]] == pytest.approx([0, 0], abs=1e-12)
    assert res["valence"] == CU_VALENCE

    e, e_ion = res["min_energies"], res["ion_reference_energy"]
    assert e_ion == -0.01
    expected_steps = (e[:-1] + e_ion - e[1:]) / CU_VALENCE
    np.testing.assert_allclose(res["voltages"], expected_steps)
    np.testing.assert_allclose(res["voltage_x"], (res["x"][:-1] + res["x"][1:]) / 2)

    # The hull average over the full range only depends on the end points.
    expected_avg = (e[0] + N_SITES * e_ion - e[-1]) / (N_SITES * CU_VALENCE)
    assert res["average_voltage"] == pytest.approx(expected_avg)
    pairs = res["voltage_pairs"]
    assert min(p["x_charge"] for p in pairs) == pytest.approx(0)
    assert max(p["x_discharge"] for p in pairs) == pytest.approx(1)
    assert res["min_voltage"] <= res["average_voltage"] <= res["max_voltage"]
    assert res["capacity_grav"] > 0

    assert res["_units"]["voltages"] == "V"
    assert res["_units"]["capacity_grav"] == "mAh/g"
    assert len(res["intercalation"]) == N_SITES + 1


def test_voltage_partial_range(Cu3Au: Structure, emt_calculator: Calculator) -> None:
    """A partial removal range samples one ion at a time within it; results describe that window only."""
    res = emt_voltage_calc(emt_calculator, removal_range=(0.0, 0.5), store_intercalation=False).calc(Cu3Au)

    np.testing.assert_array_equal(res["n_removed"], [3, 2, 1, 0])
    assert res["x_range"] == pytest.approx((0.5, 1.0))
    assert len(res["voltages"]) == 3
    assert res["mixing_energies"][[0, -1]] == pytest.approx([0, 0], abs=1e-12)
    e, e_ion = res["min_energies"], res["ion_reference_energy"]
    assert res["average_voltage"] == pytest.approx((e[0] + 3 * e_ion - e[-1]) / (3 * CU_VALENCE))
    assert min(p["x_charge"] for p in res["voltage_pairs"]) == pytest.approx(0.5)
    assert max(p["x_discharge"] for p in res["voltage_pairs"]) == pytest.approx(1.0)
    assert "intercalation" not in res


def test_voltage_window_without_full_host(Cu3Au: Structure, emt_calculator: Calculator) -> None:
    """Hull x values stay correct when the window does not include the fully intercalated host."""
    res = emt_voltage_calc(emt_calculator, removal_range=(1 / 3, 1.0)).calc(Cu3Au)

    np.testing.assert_array_equal(res["n_removed"], [6, 5, 4, 3, 2])
    assert res["x_range"] == pytest.approx((0.0, 2 / 3))
    assert min(p["x_charge"] for p in res["voltage_pairs"]) == pytest.approx(0.0)
    assert max(p["x_discharge"] for p in res["voltage_pairs"]) == pytest.approx(2 / 3)


def test_voltage_single_ion_count(Cu3Au: Structure, emt_calculator: Calculator) -> None:
    """A range holding a single ion count cannot give a voltage."""
    with pytest.raises(ValueError, match="single ion count"):
        emt_voltage_calc(emt_calculator, removal_range=(0.5, 0.5)).calc(Cu3Au)


def test_voltage_relaxed_reference(Cu3Au: Structure, Cu: Structure, emt_calculator: Calculator) -> None:
    """A Structure ion reference is relaxed with the calculator, as part of a chain after RelaxCalc."""
    chain = ChainedCalc(
        [
            RelaxCalc(emt_calculator, fmax=0.05),
            emt_voltage_calc(
                emt_calculator,
                removal_range=(0.0, 1 / 6),
                nsteps=1,
                relax=True,
                ion_reference=Cu,
                relax_calc_kwargs={"fmax": 0.05, "max_steps": 50},
            ),
        ]
    )
    res = chain.calc(Cu3Au)

    reference = RelaxCalc(emt_calculator, fmax=0.05, max_steps=50).calc(Cu)
    assert res["ion_reference_energy"] == pytest.approx(reference["energy"] / len(Cu), abs=1e-3)
    assert "a" in res  # fields from the upstream RelaxCalc are kept
    assert np.isfinite(res["average_voltage"])


def test_voltage_write_profile(Cu3Au: Structure, emt_calculator: Calculator) -> None:
    """By default a CSV with one row per ion count is written; False disables it."""
    emt_voltage_calc(emt_calculator, removal_range=(0.0, 1 / 3), write_voltage_profile=False).calc(Cu3Au)
    assert not Path("voltage_profile.csv").exists()

    emt_voltage_calc(emt_calculator, removal_range=(0.0, 1 / 3)).calc(Cu3Au)

    with Path("voltage_profile.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["n_removed"]) for r in rows] == [2, 1, 0]
    assert rows[-1]["step_voltage_V"] == ""
    assert not Path("results.json.gz").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ion_reference": "PBE-typo"}, "Unknown ion_reference"),
        ({"species": "Au"}, "no positive oxidation state|only Au"),
    ],
)
def test_voltage_bad_reference(
    Cu3Au: Structure, Cu: Structure, emt_calculator: Calculator, kwargs: dict, match: str
) -> None:
    """An unknown reference set, or a reference that is not the working-ion metal, raises ValueError."""
    if kwargs.get("species") == "Au":
        kwargs = kwargs | {"ion_reference": Cu}
    with pytest.raises(ValueError, match=match):
        emt_voltage_calc(emt_calculator, removal_range=(0.0, 0.5), **kwargs).calc(Cu3Au)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"species": "He"}, "no positive oxidation state"), ({"removal_range": (0.6, 0.4)}, "0 <= start <= stop <= 1")],
)
def test_voltage_bad_arguments(emt_calculator: Calculator, kwargs: dict, match: str) -> None:
    """A working ion without a positive valence, or an invalid removal range, is rejected."""
    with pytest.raises(ValueError, match=match):
        emt_voltage_calc(emt_calculator, **kwargs)


def test_voltage_no_working_ion(Cu: Structure, emt_calculator: Calculator) -> None:
    """A host without working-ion sites raises ValueError."""
    with pytest.raises(ValueError, match="no Li sites"):
        emt_voltage_calc(emt_calculator, species="Li", supercell=None).calc(Cu)


def test_voltage_lifepo4(LiFePO4: Structure, matpes_calculator: PESCalculator) -> None:
    """LiFePO4 with a foundation potential and the default MatPES-PBE Li reference."""
    res = VoltageCalc(
        matpes_calculator,
        nsteps=2,
        seed=42,
        relax=False,
        relax_structure=False,
        trajfile="lfp.traj",
    ).calc(LiFePO4)

    np.testing.assert_array_equal(res["n_removed"], [4, 3, 2, 1, 0])
    assert 2 < res["average_voltage"] < 4.5
    assert res["capacity_grav"] == pytest.approx(169.9, abs=0.5)  # theoretical capacity of LiFePO4
