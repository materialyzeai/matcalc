from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matcalc import RelaxCalc

if TYPE_CHECKING:
    from pathlib import Path

    from ase.filters import Filter
    from matgl.ext.ase import PESCalculator
    from pymatgen.core import Structure


def test_bad_input(Li2O: Structure, matpes_calculator: PESCalculator) -> None:
    relax_calc = RelaxCalc(
        matpes_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
    )

    with pytest.raises((ValueError, KeyError, TypeError)):
        relax_calc.calc({"bad": Li2O})

    data = list(relax_calc.calc_many([Li2O, None], allow_errors=True))
    assert data[0] is not None
    assert data[1] is None


def test_relax_calc_invalid_optimizer(
    matpes_calculator: PESCalculator,
    Li2O: Structure,
) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer='invalid', must be one of "):
        RelaxCalc(matpes_calculator, optimizer="invalid").calc(Li2O)


@pytest.mark.parametrize("input_kind", ["structure", "atoms", "dict"])
def test_static_calc_inputs(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    input_kind: str,
) -> None:
    relax_calc = RelaxCalc(matpes_calculator, relax_atoms=False, relax_cell=False)

    if input_kind == "structure":
        structure_in: Structure | Atoms | dict = Li2O
    elif input_kind == "atoms":
        structure_in = Atoms(
            symbols=[site.specie.symbol for site in Li2O],
            positions=Li2O.cart_coords,
            cell=Li2O.lattice.matrix,
            pbc=True,
        )
    else:
        structure_in = {"structure": Li2O}

    result = relax_calc.calc(structure_in)

    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"

    final_struct: Structure = result["final_structure"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"

    units = result["_units"]
    assert units["energy"] == "eV"
    assert units["forces"] == "eV/A"
    assert units["stress"] == "eV/A^3"
    assert units["volume"] == "A^3"
    assert units["a"] == units["b"] == units["c"] == "A"
    assert units["alpha"] == units["beta"] == units["gamma"] == "degree"

    if input_kind == "structure":
        expected_energy = -14.176713
        expected_forces = np.array(
            [
                [1.5045953e-07, -4.7095818e-07, -2.9371586e-07],
                [-3.2380517e-03, -2.3780954e-03, -5.1000002e-03],
                [3.2379325e-03, 2.3785862e-03, 5.1003112e-03],
            ],
            dtype=np.float32,
        )
        expected_stresses = np.array(
            [
                [0.00433428, -0.00033270, -0.00071350],
                [-0.00033270, 0.00454247, -0.00052437],
                [-0.00071350, -0.00052437, 0.00366283],
            ],
            dtype=np.float32,
        )

        assert result["energy"] == pytest.approx(expected_energy, rel=1e-1)
        assert np.allclose(result["forces"], expected_forces, atol=2e-3)
        assert np.allclose(result["stress"], expected_stresses, atol=1e-3)
    else:
        assert result["energy"] < 0
        assert result["forces"].shape == (len(Li2O), 3)
        assert result["stress"].shape == (3, 3)
        assert np.isfinite(result["forces"]).all()
        assert np.isfinite(result["stress"]).all()


@pytest.mark.parametrize(("expected_a", "expected_energy"), [(3.291072, -14.176713)])
def test_relax_calc_relax_atoms(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    expected_a: float,
    expected_energy: float,
) -> None:
    relax_calc = RelaxCalc(
        matpes_calculator,
        traj_file=f"{tmp_path}/li2o_relax_atoms.txt",
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=False,
    )
    result = relax_calc.calc(Li2O)

    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"

    final_struct: Structure = result["final_structure"]
    energy: float = result["energy"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"

    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert energy == pytest.approx(expected_energy, rel=1e-1)
    assert a == pytest.approx(expected_a, rel=1e-3)
    assert b == pytest.approx(expected_a, rel=1e-3)
    assert c == pytest.approx(expected_a, rel=1e-3)
    assert alpha == pytest.approx(60, abs=5)
    assert beta == pytest.approx(60, abs=5)
    assert gamma == pytest.approx(60, abs=5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=2)


@pytest.mark.parametrize(
    ("cell_filter", "cell_filter_kwargs", "expected_a"),
    [
        (FrechetCellFilter, None, 3.291072),
        (ExpCellFilter, None, 3.288585),
        (FrechetCellFilter, {"hydrostatic_strain": True}, 3.291072),
    ],
)
def test_relax_calc_relax_cell_variants(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
    cell_filter: Filter,
    cell_filter_kwargs: dict | None,
    expected_a: float,
) -> None:
    relax_calc = RelaxCalc(
        matpes_calculator,
        traj_file=f"{tmp_path}/li2o_relax_cell.txt",
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
        cell_filter=cell_filter,
        cell_filter_kwargs=cell_filter_kwargs,
    )
    result = relax_calc.calc(Li2O)

    for key in (
        "final_structure",
        "energy",
        "forces",
        "stress",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "volume",
    ):
        assert key in result, f"{key=} not in result"

    final_struct: Structure = result["final_structure"]
    energy: float = result["energy"]
    missing_keys = {*final_struct.lattice.params_dict} - {*result}
    assert len(missing_keys) == 0, f"{missing_keys=}"

    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert energy == pytest.approx(-14.1767, rel=1e-1)
    assert a == pytest.approx(expected_a, rel=1e-1)
    assert b == pytest.approx(expected_a, rel=1e-1)
    assert c == pytest.approx(expected_a, rel=1e-1)
    assert alpha == pytest.approx(60, abs=5)
    assert beta == pytest.approx(60, abs=5)
    assert gamma == pytest.approx(60, abs=5)
    assert final_struct.volume == pytest.approx(a * b * c / 2**0.5, abs=2)


def test_relax_calc_perturb_distance(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    tmp_path: Path,
) -> None:
    relax_calc = RelaxCalc(
        matpes_calculator,
        traj_file=f"{tmp_path}/li2o_relax_perturbed.txt",
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
        perturb_distance=0.2,
    )
    result = relax_calc.calc(Li2O)
    final_struct: Structure = result["final_structure"]
    a, b, c, alpha, beta, gamma = final_struct.lattice.parameters

    assert result["energy"] == pytest.approx(-14.176716, rel=1e-1)
    assert a == pytest.approx(3.291072, rel=1e-1)
    assert b == pytest.approx(3.291072, rel=1e-1)
    assert c == pytest.approx(3.291072, rel=1e-1)
    assert alpha == pytest.approx(60, abs=5)
    assert beta == pytest.approx(60, abs=5)
    assert gamma == pytest.approx(60, abs=5)


@pytest.mark.parametrize(
    ("fix_atoms", "fixed_indices"),
    [
        (True, [0, 1, 2]),
        ([0], [0]),
    ],
)
def test_relax_calc_fix_atoms(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
    fix_atoms: bool | list[int],
    fixed_indices: list[int],
) -> None:
    structure = Li2O.copy()
    structure.perturb(distance=0.1)

    relax_calc = RelaxCalc(
        matpes_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=False,
        fix_atoms=fix_atoms,
        fmax=0.1,
        max_steps=300,
    )
    result = relax_calc.calc(structure)
    final_struct: Structure = result["final_structure"]

    disp = np.linalg.norm(final_struct.cart_coords - structure.cart_coords, axis=1)

    for i in fixed_indices:
        assert disp[i] == pytest.approx(0.0, abs=1e-4)


def test_relax_calc_fix_symmetry(
    matpes_calculator: PESCalculator,
) -> None:
    # Use a fresh structure rather than the session-scoped Li2O fixture, since
    # other tests may have mutated it in-place.
    from pymatgen.util.testing import PymatgenTest

    reference = PymatgenTest.get_structure("Li2O")
    perturbed = reference.copy()
    perturbed.perturb(distance=0.1)

    symprec = 0.2
    spg_ref = SpacegroupAnalyzer(reference, symprec=symprec).get_space_group_number()

    relax_calc = RelaxCalc(
        matpes_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
        fix_symmetry=True,
        symprec=symprec,
        fmax=0.1,
        max_steps=300,
    )
    result = relax_calc.calc(perturbed)
    final_struct: Structure = result["final_structure"]

    spg_final = SpacegroupAnalyzer(final_struct, symprec=symprec).get_space_group_number()

    assert spg_final == spg_ref


def test_relax_calc_many(
    Li2O: Structure,
    matpes_calculator: PESCalculator,
) -> None:
    relax_calc = RelaxCalc(
        matpes_calculator,
        optimizer="FIRE",
        relax_atoms=True,
        relax_cell=True,
        cell_filter=FrechetCellFilter,
    )
    results = list(relax_calc.calc_many([Li2O] * 2))

    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None
    assert results[-1]["a"] == pytest.approx(3.291072, rel=1e-1)
