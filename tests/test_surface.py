"""Tests for the SurfaceCalc class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from matcalc import SurfaceCalc

if TYPE_CHECKING:
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_surface_calc_basic(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """
    Test the basic workflow:
      1) calc_slabs on a known structure
      2) calc on the resulting slabs
      3) Check the final results
    """
    surf_calc = SurfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_slab=True,
        fmax=0.1,
        max_steps=100,
    )

    results = list(
        surf_calc.calc_slabs(
            Si,
            miller_index=(1, 1, 1),
            min_slab_size=10.0,
            min_vacuum_size=10.0,
        )
    )
    assert len(results) == 2, "Expected two slabs for Silicon (111)."

    slab_res = results[0]

    assert "final_bulk" in slab_res
    assert "final_slab" in slab_res

    assert slab_res["bulk_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.04857103179280209, rel=1e-1)

    slab_res = results[1]

    assert slab_res["bulk_energy_per_atom"] == pytest.approx(-5.419038772583008, rel=1e-1)
    assert slab_res["surface_energy"] == pytest.approx(0.15327125170767797, rel=1e-1)


def test_surface_calc_invalid_input(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """
    If the user passes a non-dict to calc, it should raise ValueError.
    """
    surf_calc = SurfaceCalc(calculator=matpes_calculator)
    with pytest.raises(
        ValueError,
        match="For surface calculations, structure must be a dict in one of the following formats:",
    ):
        surf_calc.calc(Si)
    with pytest.raises(
        ValueError, match="For surface calculations, structure must be a dict in one of the following formats:"
    ):
        surf_calc.calc({"slab": Si})


def test_surface_calc_from_bulk_structure(Si: Structure, matpes_calculator: PESCalculator) -> None:
    """calc() with a bulk *structure* (not bulk_energy_per_atom) relaxes the bulk itself."""
    surf_calc = SurfaceCalc(
        calculator=matpes_calculator,
        relax_bulk=True,
        relax_slab=True,
        fmax=0.1,
        max_steps=100,
    )
    # Obtain a slab via calc_slabs, then feed calc a dict with a bulk *structure*
    # (no bulk_energy_per_atom) to exercise the bulk-relaxation branch of calc().
    # Note: we deliberately keep calc_slabs here (no SlabGenerator.get_slab() shortcut)
    # because slab_result is the precomputed reference we compare against below.
    slab_result = surf_calc.calc_slabs(Si, miller_index=(1, 1, 1), min_slab_size=10.0, min_vacuum_size=10.0)[0]
    # Feed the original (unrelaxed) slab, which retains surface_area; calc() will relax it.
    slab = slab_result["slab"]

    result = surf_calc.calc({"slab": slab, "bulk": Si})

    # The bulk-relaxation branch of calc() must have populated these.
    assert "final_bulk" in result
    assert "final_slab" in result

    # Consistency check: the bulk-relaxation branch of calc() must reproduce the very
    # numbers the precomputed calc_slabs path produced (slab_result: same bulk+slab+settings).
    assert result["bulk_energy_per_atom"] == pytest.approx(slab_result["bulk_energy_per_atom"], abs=1e-3)
    assert result["surface_energy"] == pytest.approx(slab_result["surface_energy"], rel=1e-2)
