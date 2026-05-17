"""Tests for QHACalc class"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

import phonopy
import pytest
from numpy.testing import assert_allclose

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
        fmax=0.1,
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
        **write_kwargs,  # type: ignore[arg-type]
    )

    result = qha_calc.calc(Li2O)

    # Test values corresponding to different scale factors
    assert result["volumes"] == pytest.approx(
        [
            23.072070780556224,
            23.7930216431222,
            24.52883695069639,
            25.279668381289056,
            26.045667612910496,
            26.826986323571,
            27.623776191280843,
        ],
        rel=1e-2,
    )

    assert result["electronic_energies"] == pytest.approx(
        [
            -14.253732681274414,
            -14.282975196838379,
            -14.298258781433105,
            -14.300981521606445,
            -14.292275428771973,
            -14.273163795471191,
            -14.244654655456543,
        ],
        abs=1e-2,
    )

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_expansion_coefficients"][ind] == pytest.approx(8.664701952239187e-05, rel=1e-1)
    assert result["gibbs_free_energies"][ind] == pytest.approx(-14.135907500154671, rel=1e-1)
    assert result["bulk_modulus_P"][ind] == pytest.approx(70.91381208222826, rel=1e-1)
    assert result["heat_capacity_P"][ind] == pytest.approx(59.2773001627165, rel=1e-1)
    assert result["gruneisen_parameters"][ind] == pytest.approx(1.694821179002536, rel=1e-1)
    units = result["_units"]
    assert units["temperatures"] == "K"
    assert units["volumes"] == "A^3"
    assert units["electronic_energies"] == "eV"
    assert units["thermal_expansion_coefficients"] == "1/K"
    assert units["gibbs_free_energies"] == "kJ/mol"
    assert units["bulk_modulus_P"] == "GPa"
    assert units["heat_capacity_P"] == "J/(K*mol)"
    assert units["gruneisen_parameters"] == "dimensionless"
    assert units["pressures"] == "GPa"
    assert len(result["scaled_structures"]) == len(result["volumes"])
    scaled_structure_volumes = [scaled_structure.volume for scaled_structure in result["scaled_structures"]]
    assert_allclose(scaled_structure_volumes, result["volumes"])
    assert result["volumes"][0] < result["volumes"][-1]
    assert result["temperatures"][0] < result["temperatures"][-1]

    # Regression #150: all temperature-indexed output arrays must have the same length
    n_temps = len(result["temperatures"])
    assert len(result["gibbs_free_energies"]) == n_temps
    assert len(result["thermal_expansion_coefficients"]) == n_temps
    assert len(result["bulk_modulus_P"]) == n_temps
    assert len(result["heat_capacity_P"]) == n_temps
    assert len(result["gruneisen_parameters"]) == n_temps

    # Only count the 8 QHA-output write_* params, not write_ha_phonon
    qha_calc_params = inspect.signature(QHACalc).parameters
    file_write_defaults = {
        key: val.default
        for key, val in qha_calc_params.items()
        if key.startswith("write_") and key != "write_ha_phonon"
    }
    assert len(file_write_defaults) == 8

    for keyword, default_path in file_write_defaults.items():
        if instance_val := write_kwargs[keyword]:
            assert os.path.isfile(str(instance_val))
        elif not default_path and not instance_val:
            assert not os.path.isfile(default_path)


def test_qha_pressure(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for QHACalc class with pressure parameter."""
    # Initialize QHACalc
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        fmax=0.05,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        pressure=10.0,
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )

    result = qha_calc.calc(Li2O)

    # Test values corresponding to different scale factors
    assert result["volumes"] == pytest.approx(
        [
            23.072070780556224,
            23.7930216431222,
            24.52883695069639,
            25.279668381289056,
            26.045667612910496,
            26.826986323571,
            27.623776191280843,
        ],
        rel=1e-2,
    )

    assert result["electronic_energies"] == pytest.approx(
        [
            -14.253732681274414,
            -14.282975196838379,
            -14.298258781433105,
            -14.300981521606445,
            -14.292275428771973,
            -14.273163795471191,
            -14.244654655456543,
        ],
        abs=1e-2,
    )

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["gibbs_free_energies"][ind] == pytest.approx(-12.604636683826666, rel=1e-1)


@pytest.mark.parametrize("relax_structure", [True, False])
def test_qha_calc_atoms(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    relax_structure: bool,
) -> None:
    """Tests QHACalc with ASE Atoms input; relax_structure=False is regression #152."""
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        fmax=0.1,
        relax_structure=relax_structure,
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )

    result = qha_calc.calc(Si_atoms)

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_expansion_coefficients"][ind] == pytest.approx(1.184e-06, rel=2e-1)
    assert len(result["volumes"]) == 7
    assert len(result["thermal_expansion_coefficients"]) == len(result["temperatures"])


def test_phonon_calc_imaginary_freq_tol(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
) -> None:

    # Initialize QHACalc and ensure no imaginaries
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        fmax=0.1,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )

    result = qha_calc.calc(Si_atoms)

    ind = result["temperatures"].tolist().index(300)
    assert result["thermal_expansion_coefficients"][ind] == pytest.approx(1.184e-06, rel=2e-1)
    assert len(result["volumes"]) == 7
    assert len(result["electronic_energies"]) == 7

    # Distorted but tol is very negative so no modes are flagged
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        fmax=100,
        imaginary_freq_tol=-100.0,
        on_imaginary_modes="error",
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )
    result = qha_calc.calc(distorted_si_atoms)
    assert len(result["volumes"]) == 7
    assert len(result["electronic_energies"]) == 7

    # Distorted with tol=-0.1, so imaginary modes are flagged → error
    distorted_si_atoms2 = Si_atoms.copy()
    distorted_si_atoms2.cell += 0.5
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        fmax=100,
        imaginary_freq_tol=-0.1,
        on_imaginary_modes="error",
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )
    with pytest.raises(ValueError, match="modes are imaginary"):
        qha_calc.calc(distorted_si_atoms2)


@pytest.mark.parametrize(
    ("fix_imaginary_attempts", "expect_log"),
    [
        (0, False),
        (1, True),
    ],
)
def test_qha_imaginary_modes_raises(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    caplog: pytest.LogCaptureFixture,
    fix_imaginary_attempts: int,
    expect_log: bool,
) -> None:
    """Test that imaginary modes with on_imaginary_modes='error' raises ValueError (with/without fix attempts)."""
    distorted_si_atoms = Si_atoms.copy()
    distorted_si_atoms.cell += 0.5
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        fmax=100,
        imaginary_freq_tol=-0.1,
        fix_imaginary_attempts=fix_imaginary_attempts,
        on_imaginary_modes="error",
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )
    with caplog.at_level(logging.INFO, logger="matcalc"), pytest.raises(ValueError, match="modes are imaginary"):
        qha_calc.calc(distorted_si_atoms)
    if expect_log:
        assert any("Imaginary mode correction attempt" in r.message for r in caplog.records)
    else:
        assert not any("Imaginary mode correction attempt" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("store_ha_phonon", "scale_factors"),
    [
        (True, [0.97, 0.99, 1.00, 1.01, 1.03]),
        (False, [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]),
    ],
)
def test_qha_store_ha_phonon(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    store_ha_phonon: bool,
    scale_factors: list[float],
) -> None:
    """Test store_ha_phonon=True populates the 'ha' key in the result; False omits it."""
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=300,
        scale_factors=scale_factors,
        fmax=0.1,
        store_ha_phonon=store_ha_phonon,
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )
    result = qha_calc.calc(Si_atoms)

    if store_ha_phonon:
        assert "ha" in result
        assert len(result["ha"]) == len(scale_factors)
        for ha_result in result["ha"]:
            assert "phonon" in ha_result
            assert isinstance(ha_result["phonon"], phonopy.Phonopy)
            assert "thermal_properties" in ha_result
            assert "frequencies" in ha_result
            assert "final_structure" in ha_result
    else:
        assert "ha" not in result


@pytest.mark.parametrize("use_custom_template", [False, True])
def test_qha_write_ha_phonon(
    Si_atoms: Atoms,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_custom_template: bool,
) -> None:
    """Test write_ha_phonon=True writes one phonopy file per scale factor (default and custom template)."""
    scale_factors = [0.97, 0.99, 1.00, 1.01, 1.03]

    if use_custom_template:
        # Use a flat path template (no subdirectories) so phonopy can write without needing mkdir
        write_ha_phonon: bool | str = str(tmp_path / "ph_{scale_factor:.4f}.yaml")
        expected_files = [tmp_path / f"ph_{sf:.4f}.yaml" for sf in scale_factors]
    else:
        write_ha_phonon = True
        expected_files = [tmp_path / f"phonon_{sf:.3f}.yaml" for sf in scale_factors]

    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=300,
        scale_factors=scale_factors,
        fmax=0.1,
        write_ha_phonon=write_ha_phonon,
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )

    if use_custom_template:
        qha_calc.calc(Si_atoms)
    else:
        monkeypatch.chdir(tmp_path)
        qha_calc.calc(Si_atoms)

    for expected_file in expected_files:
        assert expected_file.is_file(), f"Expected phonopy file not found: {expected_file}"


def test_qha_multiple_pressures(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    """Tests for QHACalc class with pressure parameter."""
    # Initialize QHACalc
    qha_calc = QHACalc(
        calculator=matpes_calculator,
        t_step=50,
        t_max=1000,
        fmax=0.05,
        scale_factors=[0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03],
        pressure=[1.0, 10.0],
        phonon_calc_kwargs={"supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
    )

    result = qha_calc.calc(Li2O)

    # Test values corresponding to different scale factors
    assert result["volumes"] == pytest.approx(
        [
            23.072070780556224,
            23.7930216431222,
            24.52883695069639,
            25.279668381289056,
            26.045667612910496,
            26.826986323571,
            27.623776191280843,
        ],
        rel=1e-2,
    )

    assert result["electronic_energies"] == pytest.approx(
        [
            -14.253732681274414,
            -14.282975196838379,
            -14.298258781433105,
            -14.300981521606445,
            -14.292275428771973,
            -14.273163795471191,
            -14.244654655456543,
        ],
        abs=1e-2,
    )

    assert len(result["qha_results"]) == 2
    assert result["qha_results"][0]["pressure"] == 1.0
    assert result["qha_results"][1]["pressure"] == 10.0

    # Test values at 300 K
    ind = result["temperatures"].tolist().index(300)
    assert result["qha_results"][0]["gibbs_free_energies"][ind] == pytest.approx(-13.974636239457125, rel=1e-1)
    assert result["qha_results"][1]["gibbs_free_energies"][ind] == pytest.approx(-12.604639023450272, rel=1e-1)
