from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

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
    from matgl.ext.ase import PESCalculator


def test_available_benchmarks() -> None:
    assert "mp-binary-pbe-elasticity-2025.1.json.gz" in get_available_benchmarks()
    assert "alexandria-binary-pbe-phonon-2025.1.json.gz" in get_available_benchmarks()


def test_get_benchmark_data() -> None:
    d = get_benchmark_data("mp-pbe-elasticity-2025.3.json.gz")
    assert len(d) > 10000
    with pytest.raises(FileNotFoundError) as _:
        get_benchmark_data("bad_url")


def test_equilibrium_benchmark(matpes_calculator: PESCalculator) -> None:
    benchmark = EquilibriumBenchmark(seed=1, n_samples=2)
    results = benchmark.run(matpes_calculator, "toy")
    assert len(results) == 2
    assert results["d_toy"].mean() == pytest.approx(0.12305854320340562, abs=1e-1)
    assert np.abs(results["Eform_toy"] - results["Eform_DFT"]).mean() == pytest.approx(0.0703378673539001, abs=1e-2)


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
    assert np.abs(results["K_vrh_toy"] - results["K_vrh_DFT"]).mean() == pytest.approx(2.9499577941620814, rel=1e-1)

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
    assert np.abs(results["CV_toy"] - results["CV_DFT"]).mean() == pytest.approx(13.510378078609543, abs=1e-1)


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
