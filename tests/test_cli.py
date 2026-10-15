from __future__ import annotations

import os
import subprocess
import sys
from collections import namedtuple
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from monty.tempfile import ScratchDir

import matcalc
from matcalc._cli import calculate_property, clear_cache, main

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure


def test_clear_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear a private cache dir so parallel workers keep the shared benchmark cache.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(matcalc.config, "BENCHMARK_DATA_DIR", cache_dir)
    args = namedtuple("args", ["yes"])  # noqa: PYI024
    a = args(yes=False)
    clear_cache(a)
    assert not matcalc.config.BENCHMARK_DATA_DIR.exists()


def test_calculate_property(LiFePO4: Structure) -> None:
    args = namedtuple("args", ["model", "property", "structure", "outfile", "verbose"])  # noqa: PYI024

    with ScratchDir(".") as _:
        cif_file = "cli_test.cif"
        LiFePO4.to(filename=cif_file)
        a = args(
            model="TensorNet",
            property="ElasticityCalc",
            structure=[cif_file],
            outfile="CLI.json",
            verbose=False,
        )
        calculate_property(a)

        assert os.path.exists(a.outfile)

        a = args(
            model="TensorNet",
            property="BadCalc",
            structure=["cli_test.cif"],
            outfile="CLI.json",
            verbose=False,
        )
        with pytest.raises(KeyError):
            calculate_property(a)

        subprocess.check_output(
            [
                "matcalc",
                "calc",
                "-p",
                "ElasticityCalc",
                "-s",
                cif_file,
                "-o",
                "cli_results.json",
            ]
        )
        subprocess.check_output(
            [
                "matcalc",
                "calc",
                "-p",
                "ElasticityCalc",
                "-s",
                cif_file,
                "-o",
                "cli_results.yaml",
                "-m",
                "pbe",
            ]
        )

        assert os.path.exists("cli_results.json")
        assert os.path.exists("cli_results.yaml")


def test_main_calc_subcommand(LiFePO4: Structure, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() dispatches through 'calc' to a non-JSON outfile (yaml branch) with verbose on."""
    with ScratchDir(".") as _:
        cif_file = "cli_main_test.cif"
        LiFePO4.to(filename=cif_file)
        outfile = "cli_main_out.yaml"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "matcalc",
                "calc",
                "-m",
                "TensorNet-MatPES-PBE-2025.2",
                "-p",
                "RelaxCalc",
                "-s",
                cif_file,
                "-o",
                outfile,
                "-v",
            ],
        )
        main()
        assert os.path.exists(outfile)


def test_main_calc_stdout(LiFePO4: Structure, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() 'calc' with no --outfile prints results to stdout (pprint branch)."""
    with ScratchDir(".") as _:
        cif_file = "cli_main_stdout.cif"
        LiFePO4.to(filename=cif_file)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "matcalc",
                "calc",
                "-m",
                "TensorNet-MatPES-PBE-2025.2",
                "-p",
                "RelaxCalc",
                "-s",
                cif_file,
            ],
        )
        main()


def test_main_clear_subcommand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() dispatches through the 'clear' subcommand (no confirmation prompt)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(matcalc.config, "BENCHMARK_DATA_DIR", cache_dir)
    monkeypatch.setattr(sys, "argv", ["matcalc", "clear"])
    main()
    assert not cache_dir.exists()
@pytest.mark.parametrize("name", ["pbe", "ORB-v2", "orb-mpa-v2"])
def test_cli_model_alias(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calculate = Mock()
    monkeypatch.setattr("matcalc._cli.calculate_property", calculate)
    monkeypatch.setattr("sys.argv", ["matcalc", "calc", "-s", "structure.cif", "-m", name])
    main()
    calculate.assert_called_once()
    expected = "TensorNet-MatPES-PBE-2025.2" if name == "pbe" else "ORB-MPA-v2"
    assert calculate.call_args.args[0].model == expected


def test_cli_unknown_model(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calculate = Mock()
    monkeypatch.setattr("matcalc._cli.calculate_property", calculate)
    monkeypatch.setattr("sys.argv", ["matcalc", "calc", "-s", "structure.cif", "-m", "TensorNet-MatPES-PBE-2025"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Unknown model" in message
    assert "Did you mean" in message
    assert "TensorNet-MatPES-PBE-2025.2" in message
    calculate.assert_not_called()
