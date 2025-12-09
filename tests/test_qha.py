"""Tests for QHACalc class"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING

import pytest

from matcalc import QHACalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    (
        "helmholtz_file",
        "volume_temp_file",
        "thermal_exp_file",
        "gibbs_file",
        "bulk_mod_file",
        "cp_numerical_file",
        "cp_polyfit_file",
        "gruneisen_file",
    ),
    [
        ("", "", "", "", "", "", "", ""),
        (
            "helmholtz.dat",
            "volume_temp.dat",
            "thermal_expansion.dat",
            "gibbs.dat",
            "bulk_mod.dat",
            "cp_numerical.dat",
            "cp_polyfit.dat",
            "gruneisen.dat",
        ),
    ],
)
def test_qha_calc(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    helmholtz_file: str,
    volume_temp_file: str,
    thermal_exp_file: str,
    gibbs_file: str,
    bulk_mod_file: str,
    cp_numerical_file: str,
    cp_polyfit_file: str,
    gruneisen_file: str,
) -> None:
    """Tests for QHACalc class."""
    # Note that the fmax is probably too high. This is for testing purposes only.

    # change dir to tmp_path
    helmholtz_volume = tmp_path / helmholtz_file if helmholtz_file else False
    volume_temperature = tmp_path / volume_temp_file if volume_temp_file else False
    thermal_expansion = tmp_path / thermal_exp_file if thermal_exp_file else False
    gibbs_temperature = tmp_path / gibbs_file if gibbs_file else False
    bulk_modulus_temperature = tmp_path / bulk_mod_file if bulk_mod_file else False
    cp_numerical = tmp_path / cp_numerical_file if cp_numerical_file else False
    cp_polyfit = tmp_path / cp_polyfit_file if cp_polyfit_file else False
    gruneisen_temperature = tmp_path / gruneisen_file if gruneisen_file else False

    write_kwargs = {
        "write_helmholtz_volume": helmholtz_volume,
        "write_volume_temperature": volume_temperature,
        "write_thermal_expansion": thermal_expansion,
        "write_gibbs_temperature": gibbs_temperature,
        "write_bulk_modulus_temperature": bulk_modulus_temperature,
        "write_heat_capacity_p_numerical": cp_numerical,
        "write_heat_capacity_p_polyfit": cp_polyfit,
        "write_gruneisen_temperature": gruneisen_temperature,
    }

    # Initialize QHACalc
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        **write_kwargs,  # type: ignore[arg-type]
    )

    result = qha_calc.calc(Li2O)

    # Test values corresponding to different scale factors
    assert result["volumes"] == pytest.approx(
        [23.07207, 23.79302, 24.52884, 25.27967, 26.04567, 26.82699, 27.62378],
        rel=1e-3,
    )

    assert result["electronic_energies"] == pytest.approx(
        [
            -14.043658256530762,
            -14.065637588500977,
            -14.07603645324707,
            -14.07599925994873,
            -14.066689491271973,
            -14.048959732055664,
            -14.023341178894043,
        ],
        abs=1e-2,
    )

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_expansion_coefficients"][ind] == pytest.approx(1.03973e-04, rel=1e-1)
    assert result["gibbs_free_energies"][ind] == pytest.approx(-14.04472, rel=1e-1)
    assert result["bulk_modulus_P"][ind] == pytest.approx(54.25954, rel=1e-1)
    assert result["heat_capacity_P"][ind] == pytest.approx(62.27455, rel=1e-1)
    assert result["gruneisen_parameters"][ind] == pytest.approx(1.688877575687573, rel=1e-1)

    qha_calc_params = inspect.signature(QHACalc).parameters
    # get all keywords starting with write_ and their default values
    file_write_defaults = {key: val.default for key, val in qha_calc_params.items() if key.startswith("write_")}
    assert len(file_write_defaults) == 8

    for keyword, default_path in file_write_defaults.items():
        if instance_val := write_kwargs[keyword]:
            assert os.path.isfile(str(instance_val))
        elif not default_path and not instance_val:
            assert not os.path.isfile(default_path)


def test_qha_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for QHACalc class."""

    # Initialize QHACalc
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
    )

    result = qha_calc.calc(Si_atoms)

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_expansion_coefficients"][ind] == pytest.approx(5.191273165438463e-06, rel=1e-1)
