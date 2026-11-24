"""Tests for the unit conversion factors"""

from __future__ import annotations

import pytest

from matcalc.units import eVA2ToJm2


def test_eVA2ToJm2() -> None:
    assert eVA2ToJm2 == pytest.approx(16.02176634)
