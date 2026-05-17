from __future__ import annotations

import os
import subprocess
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest
from monty.tempfile import ScratchDir

import matcalc
from matcalc._cli import calculate_property, clear_cache

if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_clear_cache() -> None:
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
            ]
        )

        assert os.path.exists("cli_results.json")
        assert os.path.exists("cli_results.yaml")
