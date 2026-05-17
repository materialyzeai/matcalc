"""Tests for PhononCalc class"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from matcalc import PhononCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    ("force_const_file", "band_struct_file", "dos_file", "phonon_file"),
    [("", "", "", ""), ("fc", "bs.yaml", "dos.dat", "ph.yaml")],
)
def test_phonon_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    force_const_file: str,
    band_struct_file: str,
    dos_file: str,
    phonon_file: str,
) -> None:
    """Tests for PhononCalc class"""
    # Note that the fmax is probably too high. This is for testing purposes only.

    # change dir to tmp_path
    force_constants = tmp_path / force_const_file if force_const_file else False
    band_structure_yaml = tmp_path / band_struct_file if band_struct_file else False
    total_dos_dat = tmp_path / dos_file if dos_file else False
    phonon_yaml = tmp_path / phonon_file if phonon_file else False
    write_kwargs = {
        "write_force_constants": force_constants,
        "write_band_structure": band_structure_yaml,
        "write_total_dos": total_dos_dat,
        "write_phonon": phonon_yaml,
    }
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        **write_kwargs,  # type: ignore[arg-type]
    )
    result = phonon_calc.calc(Li2O)

    # Test values at 100 K
    thermal_props = result["thermal_properties"]
    ind = thermal_props["temperatures"].tolist().index(300)
    assert thermal_props["heat_capacity"][ind] == pytest.approx(56.618323353498866, rel=1e-1)
    assert thermal_props["entropy"][ind] == pytest.approx(43.70960755706888, rel=1e-1)
    assert thermal_props["free_energy"][ind] == pytest.approx(15.620299826689202, rel=1e-1)
    units = result["_units"]
    assert units["thermal_properties.temperatures"] == "K"
    assert units["thermal_properties.heat_capacity"] == "J/(K*mol)"
    assert units["thermal_properties.entropy"] == "J/(K*mol)"
    assert units["thermal_properties.free_energy"] == "kJ/mol"
    assert units["frequencies"] == "THz"
    assert_allclose(
        result["final_structure"].lattice.abc,
        (3.291071792359756, 3.291071792359756, 3.291071792359756),
        rtol=1e-3,
    )
    assert_allclose(
        result["disp_supercells"][0].lattice.abc,
        (
            2 * result["final_structure"].lattice.abc[0],
            2 * result["final_structure"].lattice.abc[1],
            2 * result["final_structure"].lattice.abc[2],
        ),
    )
    pytest.approx(result["disp_supercells"][0].volume, result["disp_supercells"][-1].volume)

    results = list(phonon_calc.calc_many([Li2O, Li2O]))
    assert len(results) == 2

    ph_calc_params = inspect.signature(PhononCalc).parameters
    # get all keywords starting with write_ and their default values
    file_write_defaults = {key: val.default for key, val in ph_calc_params.items() if key.startswith("write_")}
    assert len(file_write_defaults) == 4

    for keyword, default_path in file_write_defaults.items():
        if instance_val := write_kwargs[keyword]:
            assert os.path.isfile(str(instance_val))
        elif not default_path and not instance_val:
            assert not os.path.isfile(default_path)


def test_phonon_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for PhononCalc class"""
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
    )
    result = phonon_calc.calc(Si_atoms)

    # Test values at 100 K
    thermal_props = result["thermal_properties"]
    ind = thermal_props["temperatures"].tolist().index(300)
    assert thermal_props["heat_capacity"][ind] == pytest.approx(43.3138042001517, rel=1e-1)


def test_phonon_calc_lattice(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        min_length=10.0,
        fmax=1,
        t_step=50,
        t_max=1000,
    )
    result = phonon_calc.calc(Si_atoms)
    assert_allclose(
        result["disp_supercells"][0].lattice.abc,
        (
            3 * result["final_structure"].lattice.abc[0],
            3 * result["final_structure"].lattice.abc[1],
            3 * result["final_structure"].lattice.abc[2],
        ),
    )
    pytest.approx(result["disp_supercells"][0].volume, result["disp_supercells"][-1].volume)


def test_phonon_calc_imaginary_freq_tol(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that imaginary frequency tolerance check works correctly."""
    # Stable structure with tolerance should pass without error
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
    )
    result = phonon_calc.calc(Si_atoms)
    assert "frequencies" in result
    assert np.min(result["frequencies"]) >= -0.1

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check raises ValueError
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
    )
    with pytest.raises(ValueError, match="modes are imaginary"):
        phonon_calc.calc(distorted_si_atoms)

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check logs a warning
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="warn",
    )
    with caplog.at_level(logging.WARNING, logger="matcalc"):
        phonon_calc.calc(distorted_si_atoms)
    assert any("modes are imaginary" in r.message for r in caplog.records)

    # Distort the structure and use a very negative tol so no imaginary modes are flagged
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-100.0,
        on_imaginary_modes="warn",
    )
    result = phonon_calc.calc(distorted_si_atoms)
    assert "frequencies" in result
    assert np.min(result["frequencies"]) < -0.1


def test_phonon_calc_fix_imaginary_attempts(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test fix_imaginary_attempts parameter for resolving imaginary modes."""
    # Stable structure which needs no fixing — correction attempt should NOT be logged
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        fix_imaginary_attempts=1,
        imaginary_freq_tol=-0.01,
        on_imaginary_modes="error",
    )
    with caplog.at_level(logging.INFO, logger="matcalc"):
        result = phonon_calc.calc(Si_atoms)
    assert np.min(result["frequencies"]) > -0.01
    assert not any("Imaginary mode correction attempt" in r.message for r in caplog.records)

    caplog.clear()

    # Distort the structure to create imaginary modes, then check that
    # the tolerance check raises ValueError
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    phonon_calc = PhononCalc(
        calculator=matpes_calculator,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=100.0,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
        fix_imaginary_attempts=1,
    )
    with caplog.at_level(logging.INFO, logger="matcalc"), pytest.raises(ValueError, match="modes are imaginary"):
        phonon_calc.calc(distorted_si_atoms)
    assert any("Imaginary mode correction attempt" in r.message for r in caplog.records)
