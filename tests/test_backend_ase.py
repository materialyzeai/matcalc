from __future__ import annotations

import ase.optimize
import pytest
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from matcalc.backend._ase import VALID_OPTIMIZERS, get_ase_optimizer, is_ase_optimizer


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
