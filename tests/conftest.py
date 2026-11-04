"""
This file defines commonly used test fixtures. These are meant to be reused in unit tests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".

Environment variables:
- MATCALC_TEST_CLEAR_CACHE: set to 1/true/yes to wipe the matgl and matcalc caches (~/.cache/matgl,
  ~/.cache/matcalc) at pytest start, forcing fresh model and benchmark downloads. CI sets this. It is off by default
  so local runs reuse cached models and concurrent pytest sessions on one machine do not delete each other's files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
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
CLEAR_CACHE_ENV = "MATCALC_TEST_CLEAR_CACHE"


def pytest_configure(config: pytest.Config) -> None:
    """Optionally reset caches, then pre-download the shared model once, in the controller only.

    Caches are cleared only when ``MATCALC_TEST_CLEAR_CACHE`` is truthy (CI sets it to exercise fresh downloads).
    Under pytest-xdist, workers would otherwise clear the caches and download the same
    model concurrently while other workers are reading them.
    """
    if hasattr(config, "workerinput"):
        return
    if os.environ.get(CLEAR_CACHE_ENV, "").strip().lower() in {"1", "true", "yes"}:
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
def lmp_gnnp() -> None:
    """Skip unless ``lmp`` is on PATH with the AdvanceSoft ``pair_style gnnp`` (ML-GNNP) that LAMMPSMDCalc uses."""
    lmp = shutil.which("lmp")
    if lmp is None:
        pytest.skip("lammps (lmp) not available")
        return
    styles = subprocess.run([lmp, "-h"], capture_output=True, text=True, check=False).stdout  # noqa: S603
    if not re.search(r"\bgnnp\b", styles):
        pytest.skip("lmp lacks pair_style gnnp (ML-GNNP), which LAMMPSMDCalc needs")


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
