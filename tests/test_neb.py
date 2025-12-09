from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from matcalc import MEP, NEBCalc, RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_neb_calc(LiFePO4: Structure, matpes_calculator: PESCalculator, tmp_path: Path) -> None:
    """Tests for NEBCalc class"""
    relax = RelaxCalc(matpes_calculator, fmax=0.5, relax_cell=False)
    image_start = LiFePO4.copy()
    image_start.remove_sites([2])
    image_start = relax.calc(image_start)["final_structure"]
    image_end = LiFePO4.copy()
    image_end.remove_sites([3])
    image_end = relax.calc(image_end)["final_structure"]
    neb_calc = NEBCalc(matpes_calculator, traj_folder=tmp_path, fmax=0.5)
    barriers = neb_calc.calc_images(image_start, image_end, n_images=5)
    assert barriers["barrier"] == pytest.approx(0.17913818359375044, rel=0.002)
    assert barriers["force"] == pytest.approx(-0.0042724609375, rel=0.002)
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
        assert orig_lat.shape == recon_lat.shape, "Reconstructed lattices should have same shape"
        assert orig_lat == pytest.approx(recon_lat), "Reconstructed lattices should match"

    # Test get_structures method
    structures = mep.get_structures()
    assert len(structures) == 7, "get_structures() should return 7 structures"

    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        NEBCalc(matpes_calculator, optimizer="invalid")
