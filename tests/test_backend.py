from __future__ import annotations

import pytest
from unittest.mock import MagicMock


def test_run_pes_calc_lammps_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When SIMULATION_BACKEND is not ASE, run_pes_calc forwards to run_lammps."""
    import matcalc.backend as backend

    sentinel = object()
    fake_run_lammps = MagicMock(return_value=sentinel)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "LAMMPS")
    monkeypatch.setattr(backend, "run_lammps", fake_run_lammps)

    result = backend.run_pes_calc("arg", key="val")
    assert result is sentinel
    fake_run_lammps.assert_called_once_with("arg", key="val")


def test_run_pes_calc_ase_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default ASE backend path forwards to run_ase."""
    import matcalc.backend as backend

    sentinel = object()
    fake_run_ase = MagicMock(return_value=sentinel)

    monkeypatch.setattr(backend, "SIMULATION_BACKEND", "ASE")
    monkeypatch.setattr(backend, "run_ase", fake_run_ase)

    result = backend.run_pes_calc("arg", key="val")
    assert result is sentinel
    fake_run_ase.assert_called_once_with("arg", key="val")
