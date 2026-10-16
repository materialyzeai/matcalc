from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import call, patch

import numpy as np
import pytest
from ase.calculators.calculator import Calculator
from pymatgen.core import Lattice, Structure

from matcalc import MEP, NEBCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import PESCalculator


def test_neb_calc(LiFePO4: Structure, matpes_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for NEBCalc class"""
    image_start = LiFePO4.copy()
    image_start.remove_sites([2])
    image_end = LiFePO4.copy()
    image_end.remove_sites([3])
    neb_calc = NEBCalc(matpes_calculator, traj_folder=tmp_path, fmax=0.5, method="aseneb")
    start_result, end_result = neb_calc.calc_endpoints(image_start, image_end, relax_calc_kwargs={"optimizer": "FIRE"})
    barriers = neb_calc.calc_images(start_result["final_structure"], end_result["final_structure"], n_images=5)
    assert barriers["barrier"] == pytest.approx(0.26219177246093794, rel=0.05)
    assert barriers["reaction_energy"] == pytest.approx(0.00128173828125, abs=1e-3)
    assert barriers["force"].shape == (5 * len(image_start), 3)
    assert np.isfinite(barriers["force"]).all()
    assert barriers["_units"] == {"barrier": "eV", "reaction_energy": "eV", "force": "eV/Å"}
    assert isinstance(barriers["mep"], MEP), "barriers['mep'] should be an MEP instance"
    mep = barriers["mep"]
    assert len(mep.labels) > 0, "MEP should have labels"
    assert len(mep.frac_coords) == 7, "MEP should have 7 images (n_images=5 + 2 endpoints)"
    assert len(mep.energies) == 7, "MEP should have 7 energies"
    # Lattices can be either a single array or a list
    if isinstance(mep.lattices, np.ndarray):
        assert mep.lattices.shape == (3, 3), "Single lattice should be a 3x3 matrix"
    else:
        assert len(mep.lattices) == 7, "MEP should have 7 lattices"
        assert all(lat.shape == (3, 3) for lat in mep.lattices), "Each lattice should be a 3x3 matrix"
    # Test as_dict method
    mep_dict = mep.as_dict()
    assert "labels" in mep_dict, "as_dict() should contain 'labels'"
    assert "images" in mep_dict, "as_dict() should contain 'images'"
    assert len(mep_dict["images"]) == 7, "as_dict() should have 7 images"
    # Check if lattice is stored at top level (same for all) or per image
    if "lattice" in mep_dict:
        # Single lattice stored at top level
        assert isinstance(mep_dict["lattice"], list), "Top-level lattice should be a list"
        for img in mep_dict["images"]:
            assert "lattice" not in img, "Images should not have lattice when stored at top level"
            assert "frac_coords" in img, "Each image should have 'frac_coords'"
            assert "energy" in img, "Each image should have 'energy'"
    else:
        # Lattice stored per image
        for img in mep_dict["images"]:
            assert "lattice" in img, "Each image should have 'lattice'"
            assert "frac_coords" in img, "Each image should have 'frac_coords'"
            assert "energy" in img, "Each image should have 'energy'"

    # Test from_dict method
    mep_reconstructed = MEP.from_dict(mep_dict)
    assert len(mep_reconstructed.labels) == len(mep.labels), "Reconstructed MEP should have same number of labels"
    assert len(mep_reconstructed.frac_coords) == len(mep.frac_coords), (
        "Reconstructed MEP should have same number of images"
    )
    assert len(mep_reconstructed.energies) == len(mep.energies), "Reconstructed MEP should have same number of energies"
    # Check that energies match
    for orig_energy, recon_energy in zip(mep.energies, mep_reconstructed.energies, strict=False):
        assert orig_energy == pytest.approx(recon_energy), "Reconstructed energies should match"
    # Check that lattices match (handle both single and list cases)
    orig_lattices_list = mep.get_lattices_list()
    recon_lattices_list = mep_reconstructed.get_lattices_list()
    assert len(orig_lattices_list) == len(recon_lattices_list), "Reconstructed MEP should have same number of lattices"
    for orig_lat, recon_lat in zip(orig_lattices_list, recon_lattices_list, strict=False):
        assert orig_lat.shape == recon_lat.shape, "Reconstructed lattices should match in shape"
        assert orig_lat == pytest.approx(recon_lat), "Reconstructed lattices should match"

    # Test get_structures method
    structures = mep.get_structures()
    assert len(structures) == 7, "get_structures() should return 7 structures"

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBCalc(matpes_calculator, optimizer="invalid")


def test_mep_structure_energy_length_mismatch() -> None:
    """MEP must raise when the number of structures and energies disagree."""
    from pymatgen.core import Lattice, Structure

    a = Structure(Lattice.from_parameters(5, 5, 5, 90, 90, 90), ["Si"], [[0.0, 0.0, 0.0]])
    b = Structure(Lattice.from_parameters(5, 5, 5, 90, 90, 90), ["Si"], [[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="Number of structures"):
        MEP([a, b], [1.0])


def test_mep_per_image_lattice_roundtrip() -> None:
    """When images carry different lattices, as_dict/from_dict keep a per-image lattice."""
    from pymatgen.core import Lattice, Structure

    struct_a = Structure(Lattice.from_parameters(5.0, 5.0, 5.0, 90, 90, 90), ["Si"], [[0.0, 0.0, 0.0]])
    struct_b = Structure(Lattice.from_parameters(5.1, 5.1, 5.1, 90, 90, 90), ["Si"], [[0.5, 0.5, 0.5]])
    mep = MEP([struct_a, struct_b], [1.0, 2.0])
    # Lattices differ, so stored as a list.
    assert isinstance(mep.lattices, list)
    assert len(mep.get_lattices_list()) == 2

    mep_dict = mep.as_dict()
    assert "lattice" not in mep_dict, "per-image lattices should not be hoisted to top level"
    for img in mep_dict["images"]:
        assert "lattice" in img, "each image should carry its own lattice"

    recon = MEP.from_dict(mep_dict)
    assert len(recon.get_structures()) == 2
    assert recon.energies == pytest.approx([1.0, 2.0])


def test_neb_calc_requires_dict(matpes_calculator: PESCalculator) -> None:
    """NEBCalc.calc must raise ValueError when structure is not a dict of images."""
    with pytest.raises(ValueError, match="For NEB calculations, structure must be a dict"):
        NEBCalc(matpes_calculator).calc("not-a-dict")


def test_neb_calc_endpoints(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    start = Li2O.copy()
    start.translate_sites([0], [0.1, 0, 0], frac_coords=False)
    end = Li2O.copy()
    end.translate_sites([0], [-0.1, 0, 0], frac_coords=False)
    neb_calc = NEBCalc(matpes_calculator, optimizer="FIRE", fmax=0.1, max_steps=20)

    start_result, end_result = neb_calc.calc_endpoints({"final_structure": start, "label": "start"}, end)

    for result in (start_result, end_result):
        assert result["final_structure"].lattice == Li2O.lattice
    assert start_result["label"] == "start"
    assert start_result["final_structure"] != start
    assert end_result["final_structure"] != end


@pytest.mark.parametrize(
    "relax_calc_kwargs",
    [None, {"fmax": 0.02, "relax_cell": True}],
)
def test_neb_calc_endpoints_settings(Si: Structure, relax_calc_kwargs: dict[str, Any] | None) -> None:
    neb_calc = NEBCalc(Calculator(), fmax=0.25, max_steps=12)
    start, end = Si.copy(), Si.copy()
    results = ({"final_structure": start, "energy": -1.0}, {"final_structure": end, "energy": -2.0})
    with patch("matcalc._relaxation.RelaxCalc") as relaxer:
        relaxer.return_value.calc.side_effect = results
        actual = neb_calc.calc_endpoints(start, end, relax_calc_kwargs=relax_calc_kwargs)

    assert actual == results
    expected = {
        "optimizer": neb_calc.optimizer,
        "fmax": 0.25,
        "max_steps": 12,
        "relax_cell": False,
        **(relax_calc_kwargs or {}),
    }
    relaxer.assert_called_once_with(neb_calc.calculator, **expected)
    assert relaxer.return_value.calc.call_args_list == [call(start), call(end)]


def test_neb_calc_images_pbc(matpes_calculator: PESCalculator) -> None:
    bulk = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]]).make_supercell(2)
    # The nearest neighbor at (0.75, 0.75, 0) hops across the cell boundary into the vacancy at the origin.
    hop = int(np.argmin(np.linalg.norm(bulk.frac_coords - [0.75, 0.75, 0], axis=1)))
    start = bulk.copy()
    start.remove_sites([0])
    end = start.copy()
    end.replace(hop - 1, "Cu", [0, 0, 0])

    def path_length(result: dict[str, Any]) -> float:
        coords = np.array([s[hop - 1].coords for s in result["mep"].get_structures()])
        return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())

    # Zero NEB steps keep the interpolated band.
    unrelaxed = NEBCalc(matpes_calculator, max_steps=0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        periodic = unrelaxed.calc_images(start, end, n_images=3)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        raw = unrelaxed.calc_images(start, end, n_images=3, pbc=False)
    assert path_length(periodic) == pytest.approx(3.6 / np.sqrt(2))
    assert path_length(raw) == pytest.approx(3 * 3.6 / np.sqrt(2))

    result = NEBCalc(matpes_calculator, fmax=0.1, max_steps=200).calc_images(start, end, n_images=3)
    assert result["converged"]
    assert path_length(result) < 3
    assert 0.3 < result["barrier"] < 1.5
