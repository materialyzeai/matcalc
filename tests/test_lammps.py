"""Tests for LAMMPSMDCalc class"""

from __future__ import annotations

import os
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

import pytest

from matcalc import LAMMPSMDCalc

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

LAMMPS_TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "matcalc" / "lammps_templates"


@pytest.mark.parametrize(
    ("ensemble", "expected_energy"),
    [
        ("nve", -10.74606),
        ("nvt", -10.81289),
        ("npt_nose_hoover", -9.75230),
    ],
)
def test_lammps_calc(
    ensemble: str,
    expected_energy: float,
    Si: Structure,
) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    """Tests for LAMMPSMDCalc class"""
    in_file = f"{ensemble}.in"
    log_file = f"{ensemble}.log"
    traj_file = f"{ensemble}.lammpstrj"

    md_calc = LAMMPSMDCalc(
        calculator="TensorNet",
        ensemble=ensemble,
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
        infile=in_file,
        logfile=log_file,
        trajfile=traj_file,
    )
    results = md_calc.calc(Si)

    assert isinstance(results, dict)
    assert "trajectory" in results
    assert "potential_energy" in results
    assert "kinetic_energy" in results
    assert "total_energy" in results
    assert results["total_energy"] == pytest.approx(expected_energy, rel=1e-1)
    assert len(results["trajectory"]) == 5

    # Verify that the log file and trajectory file have been created
    assert os.path.isfile(log_file)
    assert os.path.isfile(traj_file)


def test_lammps_atoms(Si_atoms: Atoms) -> None:
    """Tests for MDCalc class using ASE Atoms input"""
    if not which("lmp"):
        pytest.skip("lammps not available")
    with open(Path(LAMMPS_TEMPLATES_DIR / "md.template")) as f:
        script_template = f.read()

    md_calc = LAMMPSMDCalc(
        calculator="TensorNet",
        temperature=300,
        taut=0.1,
        taup=0.1,
        steps=10,
        frames=5,
    )
    md_calc.write_inputs(Si_atoms, script_template=script_template)
    results = md_calc.calc(Si_atoms)
    assert isinstance(results, dict)


def test_invalid_ensemble(Si: Structure) -> None:
    """Ensure unsupported ensemble raises ValueError"""
    with pytest.raises(
        ValueError,
        match=(
            "The specified ensemble is not supported, choose from 'nve', 'nvt',"
            r" 'nvt_nose_hoover', 'npt', 'npt_nose_hoover'."
        ),
    ):
        LAMMPSMDCalc(
            calculator="TensorNet",
            ensemble="you_know_who",
            temperature=300,
            taut=0.1,
            taup=0.1,
            steps=10,
        ).calc(Si)
