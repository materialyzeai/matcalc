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
def test_adsorption_calc_adslabs(
    Pt_bulk: Slab,
    CO2: Molecule,
    matpes_calculator: PESCalculator,
    relax_slab: bool,  # noqa: FBT001
    relax_bulk: bool,  # noqa: FBT001
    relax_adsorbate: bool,  # noqa: FBT001
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
