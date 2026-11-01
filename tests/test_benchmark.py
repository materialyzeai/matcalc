from __future__ import annotations

import itertools
import json
import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ase.calculators.emt import EMT
from huggingface_hub.errors import EntryNotFoundError
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Lattice, Structure

import matcalc.benchmark
from matcalc import PropCalc
from matcalc.benchmark import (
    BenchmarkSuite,
    CheckpointFile,
    ElasticityBenchmark,
    EquilibriumBenchmark,
    PhononBenchmark,
    SofteningBenchmark,
    get_available_benchmarks,
    get_benchmark_data,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms
    from matgl.ext.ase import PESCalculator


class _FakeCalc(PropCalc):
    """Cheap PropCalc returning fixed properties; raises for the calls whose index is in ``fail``."""

    def __init__(self, fail: tuple[int, ...] = ()) -> None:
        self.fail = fail
        self.count = 0

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        i = self.count
        self.count += 1
        if i in self.fail:
            raise RuntimeError("calculation failed")
        result = super().calc(structure)
        return result | {"formation_energy_per_atom": 0.0, "bulk_modulus_vrh": 100.0, "shear_modulus_vrh": 50.0}


def test_available_benchmarks() -> None:
    assert "mp-binary-pbe-elasticity-2025.1.json.gz" in get_available_benchmarks()
    assert "alexandria-binary-pbe-phonon-2025.1.json.gz" in get_available_benchmarks()


def test_get_benchmark_data() -> None:
    d = get_benchmark_data("mp-pbe-elasticity-2025.3.json.gz")
    assert len(d) > 10000
    with pytest.raises(EntryNotFoundError) as _:
        get_benchmark_data("bad_url")


def test_equilibrium_benchmark(matpes_calculator: PESCalculator) -> None:
    benchmark = EquilibriumBenchmark(seed=1, n_samples=2)
    results = benchmark.run(matpes_calculator, "toy")
    assert len(results) == 2
    assert results["d_toy"].mean() == pytest.approx(0.2392, abs=1e-1)
    assert np.abs(results["Eform_toy"] - results["Eform_DFT"]).mean() == pytest.approx(0.05689055202524909, abs=1e-1)


def test_elasticity_benchmark(matpes_calculator: PESCalculator) -> None:
    benchmark = ElasticityBenchmark(seed=101, n_samples=3)
    chkpt_file = "checkpoint.json"

    results = benchmark.run(
        matpes_calculator, "toy", checkpoint_file=chkpt_file, checkpoint_freq=1, delete_checkpoint_on_finish=True
    )

    # Makes sure that checkpoint file is deleted upon completion.
    assert not os.path.exists(chkpt_file)
    assert len(results) == 3
    # Compute MAE
    assert np.abs(results["K_vrh_toy"] - results["K_vrh_DFT"]).mean() == pytest.approx(10.7902612, rel=1e-1)

    benchmark = ElasticityBenchmark(benchmark_name="mp-pbe-elasticity-2025.3.json.gz", seed=0, n_samples=3)

    results = benchmark.run(
        matpes_calculator,
        "toy",
        checkpoint_file=chkpt_file,
        checkpoint_freq=1,
        delete_checkpoint_on_finish=False,
        include_full_results=True,
    )

    assert "structure" in results.columns
    assert "elastic_tensor" in results.columns

    assert os.path.exists(chkpt_file)
    results, *_ = CheckpointFile(chkpt_file).load()
    assert len(results) % 3 == 0

    os.remove(chkpt_file)


def test_phonon_benchmark(matpes_calculator: PESCalculator) -> None:
    benchmark = PhononBenchmark(seed=0, n_samples=3)
    results = benchmark.run(matpes_calculator, "toy")
    assert len(results) == 3
    assert np.abs(results["CV_toy"] - results["CV_DFT"]).mean() == pytest.approx(4.250062673114591, abs=3e-1)


def test_softening_benchmark(matpes_calculator: PESCalculator) -> None:
    benchmark = SofteningBenchmark(seed=1, n_samples=3)
    results = benchmark.run(matpes_calculator, "toy", checkpoint_freq=1, checkpoint_file="checkpoint.json")
    assert len(results) == 3
    assert "softening_scale_toy" in results


def test_benchmark_suite(matpes_calculator: PESCalculator) -> None:
    elasticity_benchmark = ElasticityBenchmark(seed=0, n_samples=2, benchmark_name="mp-pbe-elasticity-2025.3.json.gz")
    elasticity_benchmark.run(
        matpes_calculator,
        "toy1",
        checkpoint_freq=1,
        delete_checkpoint_on_finish=False,
    )
    phonon_benchmark = PhononBenchmark(seed=0, n_samples=2)
    suite = BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
    results = suite.run(
        {"toy1": matpes_calculator, "toy2": matpes_calculator}, checkpoint_freq=1, delete_checkpoint_on_finish=False
    )
    for bench, name in itertools.product(["ElasticityBenchmark", "PhononBenchmark"], ["toy1", "toy2"]):
        assert os.path.exists(f"{bench}_{name}.csv")
        os.remove(f"{bench}_{name}.csv")
    assert len(results) == 2
    assert "K_vrh_toy1" in results[0].columns
    assert "K_vrh_toy2" in results[0].columns
    assert "G_vrh_toy1" in results[0].columns
    assert "G_vrh_toy2" in results[0].columns
    assert "CV_toy1" in results[1].columns
    assert "CV_toy2" in results[1].columns


@pytest.fixture
def local_dataset(tmp_path: Path) -> Path:
    """Small local benchmark file (no network) with the fields used by all Benchmark subclasses."""
    entries = [
        {
            "mp_id": f"mp-{i}",
            "material_id": f"mp-{i}",
            "formula": "Si",
            "structure": Structure(Lattice.cubic(2.7 + 0.01 * i), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
            "bulk_modulus_vrh": 90.0,
            "shear_modulus_vrh": 60.0,
            "heat_capacity": 40.0,
            "formation_energy_per_atom": 0.0,
        }
        for i in range(10)
    ]
    path = tmp_path / "local-bench.json"
    dumpfn(entries, path)
    return path


def test_sampling_does_not_reset_global_rng(local_dataset: Path) -> None:
    random.seed(123)
    expected = random.random()  # noqa: S311
    random.seed(123)
    benchmark = ElasticityBenchmark(benchmark_name=local_dataset, seed=7, n_samples=5)
    assert random.random() == expected  # noqa: S311

    # Same selection as the historical ``random.seed(seed); random.sample(...)``.
    random.seed(7)
    assert [r["mp_id"] for r in benchmark.ground_truth] == [d["mp_id"] for d in random.sample(loadfn(local_dataset), 5)]


def test_equilibrium_benchmark_failed_structure(local_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A failed calculation yields NaN in the structure column; the fingerprint step must not crash on it.
    monkeypatch.setattr(EquilibriumBenchmark, "get_prop_calc", lambda *_args, **_kwargs: _FakeCalc(fail=(1,)))
    results = EquilibriumBenchmark(benchmark_name=local_dataset, seed=1, n_samples=2).run(None, "toy", n_jobs=1)
    assert len(results) == 2
    assert results["d_toy"].iloc[0] == pytest.approx(0, abs=1e-8)
    assert np.isnan(results["d_toy"].iloc[1])
    assert np.isnan(results["Eform_toy"].iloc[1])


def test_checkpoint_mismatch_raises(local_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ElasticityBenchmark, "get_prop_calc", lambda *_args, **_kwargs: _FakeCalc())
    chkpt = local_dataset.parent / "checkpoint.json"
    CheckpointFile(chkpt).save([{"mp_id": "mp-bogus", "bulk_modulus_vrh_toy": 1.0}])
    benchmark = ElasticityBenchmark(benchmark_name=local_dataset, seed=0, n_samples=3)
    with pytest.raises(ValueError, match="does not match this benchmark"):
        benchmark.run(None, "toy", n_jobs=1, checkpoint_file=chkpt)

    # A checkpoint that is a genuine prefix of this benchmark resumes, and a kept checkpoint is complete.
    CheckpointFile(chkpt).save([{**benchmark.ground_truth[0], "bulk_modulus_vrh_toy": 1.0}])
    results = benchmark.run(
        None, "toy", n_jobs=1, checkpoint_file=chkpt, checkpoint_freq=2, delete_checkpoint_on_finish=False
    )
    assert list(results["K_vrh_toy"]) == [1.0, 100.0, 100.0]
    saved, *_ = CheckpointFile(chkpt).load()
    assert len(saved) == 3


def test_checkpoint_save_is_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = CheckpointFile(path)
    checkpoint.save([{"a": 1}])

    def _interrupted_dumpfn(_obj: Any, fn: str | Path) -> None:
        with open(fn, "w") as f:
            f.write('[{"a": ')
        raise KeyboardInterrupt

    monkeypatch.setattr(matcalc.benchmark, "dumpfn", _interrupted_dumpfn)
    with pytest.raises(KeyboardInterrupt):
        checkpoint.save([{"a": 1}, {"a": 2}])
    assert json.loads(path.read_text()) == [{"a": 1}]


def test_prop_calc_kwargs_override(local_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # User kwargs must override the benchmark defaults instead of raising "multiple values for keyword argument".
    benchmark = PhononBenchmark(benchmark_name=local_dataset, n_samples=1)
    prop_calc = benchmark.get_prop_calc(EMT(), fmax=0.1, write_phonon="ph.yaml")
    assert prop_calc.fmax == 0.1
    assert prop_calc.write_phonon == "ph.yaml"

    monkeypatch.setattr(
        EquilibriumBenchmark, "_prepare_elemental_refs", lambda self, _calc: setattr(self, "elemental_refs", {})
    )
    prop_calc = EquilibriumBenchmark(benchmark_name=local_dataset, n_samples=1).get_prop_calc(
        EMT(), fmax=0.1, perturb_distance=None
    )
    assert prop_calc.fmax == 0.1
    assert prop_calc.perturb_distance is None


def test_phonon_process_result_temperature_grid(local_dataset: Path) -> None:
    # Heat capacity must be taken at 300 K even on a non-default temperature grid.
    benchmark = PhononBenchmark(benchmark_name=local_dataset, n_samples=1)
    temperatures = np.arange(0, 1001, 50.0)
    result = {"thermal_properties": {"temperatures": temperatures, "heat_capacity": 0.1 * temperatures}}
    assert benchmark.process_result(result, "toy")["heat_capacity_toy"] == pytest.approx(30.0)
    assert np.isnan(benchmark.process_result(None, "toy")["heat_capacity_toy"])
