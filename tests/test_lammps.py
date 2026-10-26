"""Tests for LAMMPSMDCalc class"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.outputs import parse_lammps_log

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


def test_lammps_element_mapping(Li2O: Structure, tmp_path: Path) -> None:
    # O comes first, so the composition order differs from the type order of the data file.
    structure = Li2O.get_sorted_structure(key=lambda site: site.specie.symbol != "O")
    structure.remove_oxidation_states()
    charged = structure.copy()
    charged.add_oxidation_state_by_element({"Li": 1, "O": -2})
    calc = LAMMPSMDCalc(calculator="TensorNet", relax_structure=False)
    inputs = [structure, charged, AseAtomsAdaptor.get_atoms(structure), AseAtomsAdaptor.get_atoms(charged)]
    for i, structure_in in enumerate(inputs):
        calc.write_inputs(structure_in, directory=tmp_path / str(i))
        data = LammpsData.from_file(str(tmp_path / str(i) / "md.data"), atom_style="atomic")
        script = (tmp_path / str(i) / calc.infile).read_text().splitlines()
        pair_coeff = next(line.split() for line in script if line.startswith("pair_coeff"))
        dump_modify = next(line.split() for line in script if line.startswith("dump_modify"))
        assert pair_coeff[-2:] == ["Li", "O"]
        assert dump_modify[-2:] == ["Li", "O"]
        assert [pair_coeff[-2:][int(atom_type) - 1] for atom_type in data.atoms["type"]] == ["O", "Li", "Li"]


def test_lammps_frames(Li2O: Structure, tmp_path: Path) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    calc = LAMMPSMDCalc(calculator="TensorNet", relax_structure=False, steps=25, loginterval=10, equilibration_frames=1)
    result = calc.calc(Li2O, directory=tmp_path)

    # Frames are saved at steps 0, 10 and 20; the log also reports the last step.
    thermo = parse_lammps_log(str(tmp_path / "log.lammps"))[0]
    assert list(thermo["Step"]) == [0, 10, 20, 25]
    kept = thermo[thermo["Step"].isin([10, 20])]
    assert len(result["trajectory"]) == 2
    assert result["potential_energy"] == pytest.approx(kept["PotEng"].mean())
    assert result["kinetic_energy"] == pytest.approx(kept["KinEng"].mean())
    assert result["total_energy"] == pytest.approx(kept["TotEng"].mean())
    np.testing.assert_allclose(result["final_structure"].cart_coords, result["trajectory"][-1].positions)


def test_lammps_thermo_stride(Li2O: Structure, tmp_path: Path) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    calc = LAMMPSMDCalc(calculator="TensorNet", relax_structure=False, steps=25, loginterval=10, equilibration_frames=1)
    calc.write_inputs(Li2O, directory=tmp_path)
    script = tmp_path / calc.infile
    script.write_text(re.sub(r"^thermo\s+\d+", "thermo 7", script.read_text(), flags=re.MULTILINE))
    result = calc.calc(Li2O, directory=tmp_path)

    # Frames are saved at steps 10 and 20, and the edited script outputs thermo at 0, 7, 14, 21 and 25.
    thermo = parse_lammps_log(str(tmp_path / "log.lammps"))[0]
    assert list(thermo["Step"]) == [0, 7, 14, 21, 25]
    assert result["potential_energy"] == pytest.approx(thermo["PotEng"][2])
    calc.frames = 1
    with pytest.raises(ValueError, match="No thermo output between steps 20 and 20"):
        calc.calc(Li2O, directory=tmp_path)


def test_lammps_failed_run(Li2O: Structure, tmp_path: Path) -> None:
    if not which("lmp"):
        pytest.skip("lammps not available")
    calc = LAMMPSMDCalc(calculator="TensorNet", relax_structure=False, settings={"fix_command": "fix 1 all unknown"})
    with pytest.raises(subprocess.CalledProcessError):
        calc.calc(Li2O, directory=tmp_path)
