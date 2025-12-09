"""
This file defines commonly used test fixtures. These are meant to be reused in unit tests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pymatgen.util.testing import PymatgenTest

import matcalc
from matcalc.utils import PESCalculator, to_ase_atoms

if TYPE_CHECKING:
    from collections.abc import Generator

    from ase.atoms import Atoms
    from pymatgen.core import Structure

import matgl

matgl.clear_cache(confirm=False)
matcalc.clear_cache(confirm=False)


@pytest.fixture(scope="session")
def LiFePO4() -> Structure:
    """LiFePO4 structure as session-scoped fixture (don't modify in-place,
    will affect other tests).
    """
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture(scope="session")
def Li2O() -> Structure:
    """Li2O structure as session-scoped fixture."""
    return PymatgenTest.get_structure("Li2O")


@pytest.fixture(scope="session")
def Si() -> Structure:
    """Si structure as session-scoped fixture."""
    return PymatgenTest.get_structure("Si")


@pytest.fixture(scope="session")
def SiO2() -> Structure:
    """SiO2 structure as session-scoped fixture."""
    return PymatgenTest.get_structure("SiO2")


@pytest.fixture(scope="session")
def Si_atoms() -> Atoms:
    """Si atoms as session-scoped fixture."""
    return to_ase_atoms(PymatgenTest.get_structure("Si"))


@pytest.fixture(scope="session")
def matpes_calculator() -> PESCalculator:
    """TensorNet calculator as session-scoped fixture."""
    return matcalc.load_fp("TensorNet-MatPES-PBE-v2025.1-PES")


@pytest.fixture(autouse=True)
def setup_teardown() -> Generator:
    """
    Fixture method for setting up and tearing down temporary directory environment for testing.

    Returns:
        Generator: A generator yielding the path of the temporary directory.
    """
    initial_files = os.listdir()
    yield
    for f in os.listdir():
        if f not in initial_files:
            print(f"Deleting generated file: {f}")  # noqa:T201
            os.remove(f)
