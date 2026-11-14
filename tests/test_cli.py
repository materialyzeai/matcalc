from __future__ import annotations

import os
import subprocess
import sys
from collections import namedtuple
from typing import TYPE_CHECKING, Any

import pytest
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pymatgen.core import Structure

import matcalc
from matcalc._cli import CLI_CALCS, calculate_property, clear_cache, main

if TYPE_CHECKING:
    from pathlib import Path


def test_clear_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear a private cache dir so parallel workers keep the shared benchmark cache.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(matcalc.config, "BENCHMARK_DATA_DIR", cache_dir)
    args = namedtuple("args", ["yes"])  # noqa: PYI024
    a = args(yes=True)
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
        # Structures round-trip through the yaml output instead of being written as their str() summaries.
        assert isinstance(loadfn(outfile)[0]["final_structure"], Structure)


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
    monkeypatch.setattr(sys, "argv", ["matcalc", "clear", "-y"])
    main()
    assert not cache_dir.exists()


def test_cli_unknown_model() -> None:
    result = subprocess.run(
        ["matcalc", "calc", "-p", "ElasticityCalc", "-s", "structure.cif", "-m", "TensorNet-MatPES-PBE-2025"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unknown model" in result.stderr
    assert "Did you mean" in result.stderr
    assert "TensorNet-MatPES-PBE-2025.2" in result.stderr


def test_cli_removed_model() -> None:
    """A removed model name exits with the removal reason and its replacement."""
    result = subprocess.run(
        ["matcalc", "calc", "-s", "structure.cif", "-m", "QET-MatQ"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "'QET-MatQ' was removed" in result.stderr
    assert "QET-MatPES-r2SCAN-2025.2" in result.stderr


@pytest.mark.parametrize("answer", ["y", "n"])
def test_main_clear_prompts_without_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Without -y/--yes, 'matcalc clear' asks for confirmation and honors the answer."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(matcalc.config, "BENCHMARK_DATA_DIR", cache_dir)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "argv", ["matcalc", "clear"])
    main()
    assert len(prompts) == 1
    assert cache_dir.exists() is (answer == "n")


def test_main_no_subcommand(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Running 'matcalc' with no sub-command prints usage and exits with code 2 instead of an AttributeError."""
    monkeypatch.setattr(sys, "argv", ["matcalc"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_calculate_property_passes_model_name(LiFePO4: Structure, monkeypatch: pytest.MonkeyPatch) -> None:
    """PropCalcs receive the model name, which their calculator setter resolves (directly to LAMMPS on that backend)."""
    received: list[Any] = []

    class _Recorder:
        def __init__(self, calculator: Any) -> None:
            received.append(calculator)

        def calc(self, structure: Structure) -> dict:
            return {"final_structure": structure}

    monkeypatch.setitem(CLI_CALCS, "MDCalc", _Recorder)
    args = namedtuple("args", ["model", "property", "structure", "outfile", "verbose"])  # noqa: PYI024
    with ScratchDir("."):
        LiFePO4.to(filename="cli_name.cif")
        calculate_property(
            args(
                model="TensorNet-MatPES-PBE-2025.2",
                property="MDCalc",
                structure=["cli_name.cif"],
                outfile="cli_name.json",
                verbose=False,
            )
        )
    assert received == ["TensorNet-MatPES-PBE-2025.2"]


def test_main_default_model_is_r2scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without -m, 'matcalc calc' uses the r2SCAN TensorNet foundation potential, not a PBE one."""
    received: list[str] = []
    monkeypatch.setattr("matcalc._cli.calculate_property", lambda args: received.append(args.model))
    monkeypatch.setattr(sys, "argv", ["matcalc", "calc", "-s", "structure.cif"])
    main()
    assert received == ["TensorNet-MatPES-r2SCAN-2025.2"]
