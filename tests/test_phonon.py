"""Tests for PhononCalc class"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

import pytest

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
    assert thermal_props["heat_capacity"][ind] == pytest.approx(58.42898, rel=1e-1)
    assert thermal_props["entropy"][ind] == pytest.approx(49.37746, rel=1e-1)
    assert thermal_props["free_energy"][ind] == pytest.approx(13.24547, rel=1e-1)

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
