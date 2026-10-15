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

from typing import TYPE_CHECKING

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.util.testing import PymatgenTest

import matcalc
from matcalc.utils import PESCalculator, to_ase_atoms

if TYPE_CHECKING:
    from pathlib import Path

    from ase.atoms import Atoms

import matgl

MATPES_MODEL = "TensorNet-MatPES-PBE-2025.2"


def pytest_configure(config: pytest.Config) -> None:
    """Reset caches and pre-download the shared model once, in the controller only.

    Under pytest-xdist, workers would otherwise clear the caches and download the same
    model concurrently while other workers are reading them.
    """
    if hasattr(config, "workerinput"):
        return
    matgl.clear_cache(confirm=False)
    matcalc.clear_cache(confirm=False)
    matcalc.load_fp(MATPES_MODEL)


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
def Cu() -> Structure:
    """Conventional fcc Cu structure as session-scoped fixture."""
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])


@pytest.fixture
def Si_atoms() -> Atoms:
    """Si atoms, rebuilt per test since calculators and MD runs mutate Atoms in place."""
    return to_ase_atoms(PymatgenTest.get_structure("Si"))


@pytest.fixture(scope="session")
def matpes_calculator() -> PESCalculator:
    """TensorNet calculator as session-scoped fixture."""
    return matcalc.load_fp(MATPES_MODEL)


@pytest.fixture(autouse=True)
def setup_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in its own temporary directory.

    Files written to the working directory are isolated per test, so parallel
    pytest-xdist workers cannot overwrite or delete each other's outputs.
    """
    monkeypatch.chdir(tmp_path)
