from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ase.optimize
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.optimize.optimize import Optimizer

from matcalc.backend._ase import VALID_OPTIMIZERS, get_ase_optimizer, is_ase_optimizer, run_ase

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms


def test_get_ase_optimizer() -> None:
    for name in dir(ase.optimize):
        if is_ase_optimizer(name):
            optimizer = get_ase_optimizer(name)
            assert issubclass(optimizer, Optimizer)
            same_optimizer = get_ase_optimizer(optimizer)  # test ASE Optimizer classes are returned as-is
            assert optimizer is same_optimizer

    for optimizer in ("whatever", 42):
        with pytest.raises(ValueError, match=f"Unknown {optimizer=}") as exc:
            get_ase_optimizer(optimizer)
        assert str(exc.value) == f"Unknown {optimizer=}, must be one of {VALID_OPTIMIZERS}"


def test_is_ase_optimizer() -> None:
    assert is_ase_optimizer(ase.optimize.BFGS)
    assert is_ase_optimizer(Optimizer)
    assert not is_ase_optimizer(Calculator)

    for name in ("whatever", 42, -3.14):
        assert not is_ase_optimizer(name)


@pytest.fixture
def rattled_cu() -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(stdev=0.05, seed=0)
    return atoms


def _max_step_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [rec for rec in caplog.records if "Maximum steps reached" in rec.getMessage()]


def test_run_ase_no_warning_when_converged_on_last_step(
    rattled_cu: Atoms, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Converging on exactly the last allowed step is not a max-steps failure."""
    traj_file = str(tmp_path / "relax.traj")
    run_ase(rattled_cu, EMT(), relax_atoms=True, max_steps=500, fmax=0.01, traj_file=traj_file)
    n_steps = len(Trajectory(traj_file)) - 1  # initial frame + one per step
    assert n_steps > 1

    caplog.set_level(logging.WARNING, logger="matcalc.backend._ase")
    result = run_ase(rattled_cu, EMT(), relax_atoms=True, max_steps=n_steps, fmax=0.01)
    assert np.linalg.norm(result.forces, axis=1).max() <= 0.01
    assert not _max_step_warnings(caplog)


def test_run_ase_warns_once_when_not_converged(rattled_cu: Atoms, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="matcalc.backend._ase")
    run_ase(rattled_cu, EMT(), relax_atoms=True, relax_cell=True, max_steps=1, fmax=1e-6)
    warnings = _max_step_warnings(caplog)
    assert len(warnings) == 1
    assert "Max|F|" in warnings[0].getMessage()


def test_run_ase_reports_convergence(rattled_cu: Atoms) -> None:
    """``converged`` carries the optimizer's verdict for relaxations and is None for static calculations."""
    assert run_ase(rattled_cu, EMT()).converged is None
    assert run_ase(rattled_cu, EMT(), relax_atoms=True, max_steps=500, fmax=0.05).converged is True
    assert run_ase(rattled_cu, EMT(), relax_atoms=True, max_steps=1, fmax=1e-6).converged is False


def test_run_ase_converged_includes_cell(rattled_cu: Atoms) -> None:
    """A strained cell with fixed atoms has zero forces but is not converged until the stress is relaxed."""
    strained = rattled_cu.copy()
    strained.set_cell(strained.cell * 1.05, scale_atoms=True)
    strained.set_constraint(FixAtoms(indices=list(range(len(strained)))))
    stopped = run_ase(strained, EMT(), relax_atoms=True, relax_cell=True, max_steps=1, fmax=0.05)
    assert np.abs(stopped.forces).max() == 0
    assert stopped.converged is False
    relaxed = run_ase(strained, EMT(), relax_atoms=True, relax_cell=True, max_steps=500, fmax=0.05)
    assert relaxed.converged is True
