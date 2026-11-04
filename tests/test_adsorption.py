"""Tests for the AdsorptionCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.surface import SlabGenerator

from matcalc import AdsorptionCalc
from matcalc.utils import to_ase_atoms

if TYPE_CHECKING:
    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core.surface import Slab


@pytest.fixture(scope="module")
def Pt_bulk() -> Structure:
    """Pt bulk as module-scoped fixture."""
    return Structure(
        Lattice.cubic(3.924), ["Pt"] * 4, np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    )


@pytest.fixture(scope="module")
def Pt_slab(Pt_bulk: Structure) -> Slab:
    """Pt slab as module-scoped fixture."""
    slabgen = SlabGenerator(Pt_bulk, (1, 1, 1), 10, 10)
    return slabgen.get_slab()


@pytest.fixture(scope="module")
def Pt_slab_atoms(Pt_slab: Slab) -> Atoms:
    """Si slab as session-scoped fixture."""
    return to_ase_atoms(Pt_slab)


@pytest.fixture(scope="module")
def CO2() -> Molecule:
    """CO2 molecule as module-scoped fixture."""
    return Molecule(
        ["C", "O", "O"],
        np.array([[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]),
    )


@pytest.fixture(scope="module")
def Pt_adslab(Pt_slab: Slab, CO2: Molecule) -> Structure:
    """Pt slab with CO2 adsorbed as module-scoped fixture."""
    asf = AdsorbateSiteFinder(Pt_slab)
    adsite = asf.find_adsorption_sites()["all"][0]
    return asf.add_adsorbate(CO2, adsite)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ({"adsorbate_energy": None}, (-5.8437, -22.68405)),
        ({"adsorbate_energy": 4.0}, (-5.8437, 4.0)),
        ({"adsorbate_energy": 4.0, "slab_energy": 20.0}, (1.0, 4.0)),
    ],
)
@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_slab_inputs(
    Pt_slab: Slab,
    CO2: Molecule,
    Pt_adslab: Structure,
    matpes_calculator: PESCalculator,
    test_input: dict,
    expected: tuple[float, float],
) -> None:
    """Test adsorption calculation with different slab inputs."""

    structure = {
        "slab": Pt_slab,
        "adsorbate": CO2,
        "adslab": Pt_adslab,
    } | test_input

    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
    )

    results = ad_calc.calc(structure)

    assert (results["slab_energy_per_atom"]) == pytest.approx(expected[0], rel=1e-1)
    assert (results["adsorbate_energy"]) == pytest.approx(expected[1], rel=1e-1)
    assert "adsorption_energy" in results


@pytest.mark.parametrize("relax_slab", [True, False])
@pytest.mark.parametrize("relax_bulk", [True, False])
@pytest.mark.parametrize("relax_adsorbate", [True, False])
@pytest.mark.parametrize("min_area_extent", [None, (8.0, 6.0)])
@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_adslabs(
    Pt_bulk: Slab,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
    relax_slab: bool,
    relax_bulk: bool,
    relax_adsorbate: bool,
    min_area_extent: tuple[float, float] | None,
) -> None:
    """Test adsorption calculation over multiple adsorption sites."""
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=relax_slab,
        relax_bulk=relax_bulk,
        relax_adsorbate=relax_adsorbate,
    )

    results = ad_calc.calc_adslabs(
        adsorbate=CO2,
        adsorbate_energy=1.0,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        inplane_supercell=(3, 3),
        min_area_extent=min_area_extent,
        slab_gen_kwargs={
            "center_slab": True,
        },
        adsorption_sites="hollow",
        fixed_height=4.0,
        dry_run=True,
    )

    slab = results[0]["slab"]
    assert len(results) == 2, "There are two hollow sites on Pt (111) surface."
    assert "adslab" in results[0]
    assert results[0]["adsorption_site"] == "hollow"
    assert results[1]["adsorption_site_index"] == 1
    assert len(results[0]["adslab"]) == 54 + 3
    assert "Pt" in results[0]["adslab"].symbol_set
    assert "C" in results[0]["adslab"].symbol_set
    assert "O" in results[0]["adslab"].symbol_set
    assert slab.lattice.a >= min_area_extent[0] if min_area_extent else True
    assert (
        (
            np.abs(np.linalg.norm(np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])))
            >= min_area_extent[0] * min_area_extent[1]
        )
        if min_area_extent
        else True
    )
    assert "selective_dynamics" in results[0]["adslab"].site_properties
    assert results[0]["adsorbate_energy"] == pytest.approx(1.0, rel=1e-1)
    if relax_slab:
        assert results[0]["slab_energy_per_atom"] == pytest.approx(-6.016221, rel=1.0e-1)


@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_adslabs_all_sites(
    Pt_bulk: Structure,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
) -> None:
    """``adsorption_sites='all'`` expands the site dict (minus the 'all' key itself)."""
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
    )
    results = ad_calc.calc_adslabs(
        adsorbate=CO2,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites="all",
        fixed_height=4.0,
        dry_run=True,
    )
    assert len(results) >= 1
    # The synthetic 'all' key is popped before building adslabs.
    assert all(r["adsorption_site"] != "all" for r in results)


@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_adslabs_invalid_site(
    Pt_bulk: Structure,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
) -> None:
    """An unknown named site raises KeyError with a helpful message."""
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
    )
    with pytest.raises(KeyError, match="must be one"):
        ad_calc.calc_adslabs(
            adsorbate=CO2,
            bulk=Pt_bulk,
            miller_index=(1, 1, 1),
            min_slab_size=10.0,
            min_vacuum_size=10.0,
            adsorption_sites="nonexistent",
            fixed_height=4.0,
            dry_run=True,
        )


def test_adsorption_calc_adslabs_runs_calc_many(
    Pt_bulk: Structure,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
) -> None:
    """With ``dry_run=False``, ``calc_adslabs`` runs the underlying relaxations.

    This is the only test that runs real (non-dry-run) relaxations, so it keeps
    ``max_steps`` small to bound runtime while still exercising ``calc_many``.
    """
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
        fmax=0.1,
        max_steps=5,
    )
    results = ad_calc.calc_adslabs(
        adsorbate=CO2,
        adsorbate_energy=1.0,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites="hollow",
        fixed_height=4.0,
        slab_gen_kwargs={"center_slab": True},
    )
    # Sturdier than a hard-coded fcc/hcp site count (which varies with the
    # surface): require a non-empty result, all at the requested site, and a
    # finite adsorption energy.
    assert len(results) >= 1
    assert all(r["adsorption_site"] == "hollow" for r in results)
    for r in results:
        assert "adsorption_energy" in r
        assert np.isfinite(r["adsorption_energy"])


def test_adsorption_calc_requires_dict(
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` on a non-dict input raises TypeError."""
    ad_calc = AdsorptionCalc(calculator=matpes_calculator)
    with pytest.raises(TypeError):
        ad_calc.calc("not a dict")  # type: ignore[arg-type]


def test_adsorption_calc_requires_adslab(
    Pt_slab: Slab,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
) -> None:
    """``calc`` on a dict missing the 'adslab' key raises ValueError."""
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
    )
    with pytest.raises(ValueError, match="adslab"):
        ad_calc.calc({"slab": Pt_slab, "adsorbate": CO2})


@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_adslabs_custom_site_dict(
    Pt_bulk: Structure,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
) -> None:
    """``adsorption_sites`` as a custom dict (neither 'all' nor a known name).

    The custom site coordinate is taken from a dry-run built with the same
    ``bulk``/``miller_index``/slab settings, so it lives in the exact Cartesian
    frame ``calc_adslabs`` builds the internal slab in; a coordinate from a
    *separate* slab object (a different ``SlabGenerator``) would not match the
    internal surface and could place the adsorbate off the slab.
    """
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator,
        relax_slab=False,
        relax_bulk=False,
        relax_adsorbate=False,
        fmax=0.1,
        max_steps=5,
    )
    base = ad_calc.calc_adslabs(
        adsorbate=CO2,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites="hollow",
        fixed_height=4.0,
        dry_run=True,
    )
    coords = base[0]["adsorption_site_coord"]
    results = ad_calc.calc_adslabs(
        adsorbate=CO2,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites={"hollow": [coords]},
        fixed_height=4.0,
        dry_run=True,
    )
    assert len(results) == 1
    assert results[0]["adsorption_site"] == "hollow"
    # The custom site must rest on the surface: the adsorbate (the last
    # ``len(CO2)`` atoms) sits above the top slab layer of the adslab, rather
    # than off the slab in empty vacuum.
    adslab = results[0]["adslab"]
    top_slab_z = float(np.max(results[0]["final_slab"].cart_coords[:, 2]))
    ads = adslab.cart_coords[len(adslab) - len(CO2) :]
    assert float(np.min(ads[:, 2])) > top_slab_z


def test_adsorption_calc_slab_relaxed_energy(Pt_slab: Slab, matpes_calculator: PESCalculator) -> None:
    """With relax_slab, the slab energy is that of the relaxed slab, not of the input slab."""
    ad_calc = AdsorptionCalc(calculator=matpes_calculator, relax_slab=True, max_steps=100)
    res = ad_calc.calc_slab(Pt_slab)

    def energy(structure: Structure) -> float:
        atoms = to_ase_atoms(structure)
        atoms.calc = matpes_calculator
        return float(atoms.get_potential_energy())

    assert res["slab_energy"] == pytest.approx(energy(res["final_slab"]), abs=1e-3)
    assert res["slab_energy"] < energy(Pt_slab) - 1e-2
    assert res["slab_energy_per_atom"] == pytest.approx(res["slab_energy"] / len(Pt_slab))


@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_adslabs_keeps_slab_gen_kwargs(
    Pt_bulk: Structure, CO2: Molecule, matpes_calculator: PESCalculator
) -> None:
    """calc_adslabs does not mutate the caller's slab_gen_kwargs, so reusing it for another Miller index does not
    carry over the max_normal_search default of the first one."""
    ad_calc = AdsorptionCalc(calculator=matpes_calculator, relax_slab=False, relax_bulk=False, relax_adsorbate=False)
    slab_gen_kwargs = {"center_slab": True}
    ad_calc.calc_adslabs(
        adsorbate=CO2,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites="hollow",
        slab_gen_kwargs=slab_gen_kwargs,
        dry_run=True,
    )
    assert slab_gen_kwargs == {"center_slab": True}


@pytest.mark.ase_only(reason="isolated adsorbate molecules are not periodic")
def test_adsorption_calc_reuses_slab_and_adsorbate(
    Pt_bulk: Structure,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calc does not recompute the clean slab and adsorbate that calc_adslabs already computed once for all sites."""
    ad_calc = AdsorptionCalc(
        calculator=matpes_calculator, relax_slab=False, relax_bulk=False, relax_adsorbate=False, max_steps=5
    )
    adslabs = ad_calc.calc_adslabs(
        adsorbate=CO2,
        bulk=Pt_bulk,
        miller_index=(1, 1, 1),
        min_slab_size=10.0,
        min_vacuum_size=10.0,
        adsorption_sites="hollow",
        fixed_height=4.0,
        dry_run=True,
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("recomputed")

    monkeypatch.setattr(ad_calc, "calc_slab", fail)
    monkeypatch.setattr(ad_calc, "calc_adsorbate", fail)
    for adslab in adslabs:
        res = ad_calc.calc(adslab)
        assert res["slab_energy_per_atom"] == pytest.approx(adslab["slab_energy"] / len(adslab["slab"]))
        assert res["adsorption_energy"] == pytest.approx(
            res["adslab_energy"] - adslab["slab_energy"] - adslab["adsorbate_energy"]
        )


def test_adsorption_calc_bulk_relaxation(Pt_bulk: Structure, matpes_calculator: PESCalculator) -> None:
    """The bulk goes through PropCalc._check_and_prelax (an unconverged relaxation raises), and calc_bulk
    records the documented final_bulk / bulk_energy / n_bulk_atoms attributes."""
    rattled = Pt_bulk.copy().perturb(0.2, seed=0)
    with pytest.raises(RuntimeError, match="Pre-relaxation did not converge"):
        AdsorptionCalc(calculator=matpes_calculator, max_steps=1).calc_bulk(rattled)

    ad_calc = AdsorptionCalc(calculator=matpes_calculator, max_steps=100)
    bulk_opt = ad_calc.calc_bulk(Pt_bulk)
    assert ad_calc.final_bulk == bulk_opt["final_structure"]
    assert ad_calc.bulk_energy == pytest.approx(bulk_opt["energy"])
    assert ad_calc.n_bulk_atoms == len(Pt_bulk)
