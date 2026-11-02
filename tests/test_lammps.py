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
    ("name", "expected_model"),
    [
        ("TensorNet-MatPES-PBE-2025.2", "TensorNet-PES-MatPES-PBE-2025.2"),
        ("pbe", "TensorNet-PES-MatPES-PBE-2025.2"),
        ("tensornet-pes-matpes-pbe-2025.2", "TensorNet-PES-MatPES-PBE-2025.2"),
        ("m3gnet", "M3GNet-PES-MatPES-PBE-2025.2"),
        ("M3GNet-MatPES-r2SCAN-2025.2", "M3GNet-PES-MatPES-r2SCAN-2025.2"),
    ],
)
def test_lammps_model_names(name: str, expected_model: str) -> None:
    calc = LAMMPSMDCalc(name, relax_structure=False)
    assert calc.model_name == expected_model
    assert calc.gnnp_type == "matgl"


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


def test_lammps_local_matgl_model(tmp_path: Path) -> None:
    import matgl

    model_dir = tmp_path / "finetuned"
    matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2").save(model_dir)
    calc = LAMMPSMDCalc(str(model_dir), relax_structure=False)
    assert calc.model_name == str(model_dir)
    assert calc.gnnp_type == "matgl"


def test_lammps_removed_model() -> None:
    with pytest.raises(ValueError, match="was removed"):
        LAMMPSMDCalc("M3GNet-MatPES-PBE-2025.1", relax_structure=False)


def test_lammps_parse_outputs(Li2O: Structure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dump and log parsing, exercised without LAMMPS by faking the ``lmp`` run."""
    from pymatgen.io.ase import MSONAtoms

    # Triclinic LAMMPS cell a = (4, 0, 0), b = (1, 5, 0), c = (0.5, 0, 6), written with bounding-box bounds.
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.0, 6.0]])
    symbols = ["O", "Li", "Li"]
    positions = {step: np.arange(9, dtype=float).reshape(3, 3) / 10 + step for step in (0, 10, 20)}
    energies = {0: (-15.0, 0.10), 10: (-14.8, 0.20), 20: (-14.6, 0.40)}

    def fake_run(args: list[str], *, check: bool, cwd: Path) -> None:
        assert args == ["lmp", "-in", calc.infile]
        assert check
        dump = []
        for step, pos in positions.items():
            dump += [
                "ITEM: TIMESTEP",
                str(step),
                "ITEM: NUMBER OF ATOMS",
                "3",
                "ITEM: BOX BOUNDS xy xz yz pp pp pp",
                "0.0 5.5 1.0",
                "0.0 5.0 0.5",
                "0.0 6.0 0.0",
                "ITEM: ATOMS id element x y z",
                *(f"{i + 1} {sym} {x} {y} {z}" for i, (sym, (x, y, z)) in enumerate(zip(symbols, pos, strict=True))),
            ]
        (Path(cwd) / calc.trajfile).write_text("\n".join(dump) + "\n")
        log = [
            "Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes",
            "Step Time CPU PotEng KinEng TotEng Temp Press Volume Density",
            *(f"{step} {step / 1000} 0 {pe} {ke} {pe + ke} 300 0 120 1" for step, (pe, ke) in energies.items()),
            "Loop time of 0.1 on 1 procs for 20 steps with 3 atoms",
        ]
        (Path(cwd) / calc.logfile).write_text("\n".join(log) + "\n")

    calc = LAMMPSMDCalc(calculator="TensorNet", relax_structure=False, steps=20, loginterval=10, frames=2)
    monkeypatch.setattr("matcalc._lammps.subprocess.run", fake_run)
    result = calc.calc(Li2O, directory=tmp_path)

    assert len(result["trajectory"]) == 2
    for atoms, step in zip(result["trajectory"], (10, 20), strict=True):
        assert isinstance(atoms, MSONAtoms)
        assert atoms.get_chemical_symbols() == symbols
        assert atoms.pbc.all()
        np.testing.assert_allclose(atoms.cell[:], cell)
        np.testing.assert_allclose(atoms.positions, positions[step])
    assert result["potential_energy"] == pytest.approx(-14.7)
    assert result["kinetic_energy"] == pytest.approx(0.3)
    assert result["total_energy"] == pytest.approx(-14.4)
    final = result["final_structure"]
    assert [site.species_string for site in final] == symbols
    np.testing.assert_allclose(final.cart_coords, positions[20])
    np.testing.assert_allclose(final.lattice.matrix, cell)
