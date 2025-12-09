"""Tests for Phonon3Calc class"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from matcalc import Phonon3Calc

if TYPE_CHECKING:
    from pathlib import Path

    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    ("phonon3_file", "write_kappa"),
    [("", False), ("ph3.yaml", True)],
)
def test_phonon3_calc(
    Si: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    phonon3_file: str,
    *,
    write_kappa: bool,
) -> None:
    """Tests for Phonon3Calc class"""
    # Note: fmax is set relatively high for testing purposes only.

    # Change directory to tmp_path to ensure that generated files are created there.
    write_phonon3_path = tmp_path / phonon3_file if phonon3_file else False
    mesh_numbers = (10, 10, 10)

    phonon3_calc = Phonon3Calc(
        calculator=matpes_calculator,
        fc2_supercell=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fc3_supercell=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        mesh_numbers=mesh_numbers,
        disp_kwargs={"distance": 0.03},
        thermal_conductivity_kwargs={"is_isotope": True, "conductivity_type": "wigner"},
        write_phonon3=write_phonon3_path,
        write_kappa=write_kappa,
    )

    result = phonon3_calc.calc(Si)
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_conductivity"][ind] == pytest.approx(76.02046300874582, rel=1e-1)

    if write_phonon3_path:
        assert os.path.isfile(str(write_phonon3_path))
    else:
        assert not os.path.isfile(str(write_phonon3_path))

    if write_kappa:
        assert os.path.isfile(f"kappa-m{''.join(map(str, mesh_numbers))}.hdf5")
    else:
        assert not os.path.isfile(f"kappa-m{''.join(map(str, mesh_numbers))}.hdf5")


def test_phonon3_calc_atoms(
    Si_atoms: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for Phonon3Calc class"""
    mesh_numbers = (10, 10, 10)

    phonon3_calc = Phonon3Calc(
        calculator=matpes_calculator,
        fc2_supercell=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fc3_supercell=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        fmax=0.1,
        t_step=50,
        t_max=1000,
        mesh_numbers=mesh_numbers,
        disp_kwargs={"distance": 0.03},
        thermal_conductivity_kwargs={"is_isotope": True, "conductivity_type": "wigner"},
    )

    result = phonon3_calc.calc(Si_atoms)
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_conductivity"][ind] == pytest.approx(76.02046300874582, rel=1e-1)
