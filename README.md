<h1 align="center">
  <img src="https://raw.githubusercontent.com/materialsvirtuallab/matcalc/refs/heads/main/docs/assets/matcalc.png"
width="200" alt="MatCalc" style="vertical-align: middle;" /><br>
</h1>

[![Test](https://github.com/materialsvirtuallab/matcalc/workflows/Test/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Test/badge.svg)
[![Lint](https://github.com/materialsvirtuallab/matcalc/workflows/Lint/badge.svg)](https://github.com/materialsvirtuallab/matcalc/workflows/Lint/badge.svg)
[![codecov](https://codecov.io/gh/materialsvirtuallab/matcalc/branch/main/graph/badge.svg?token=OR7Z9WWRRC)](https://codecov.io/gh/materialsvirtuallab/matcalc)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/matcalc?logo=pypi&logoColor=white)](https://pypi.org/project/matcalc?logo=pypi&logoColor=white)
[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matcalc)](https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE)

## Introduction

MatCalc is a Python library for calculating and benchmarking material properties from the potential energy surface
(PES). The PES can come from DFT or, more commonly, from machine learning interatomic potentials (MLIPs).

Calculating material properties often requires involved setups of various simulation codes. The
goal of MatCalc is to provide a simplified, consistent interface to access these properties with any
parameterization of the PES.

MatCalc is part of the MatML ecosystem, which includes the [MatGL] (Materials Graph Library) and [MAML] (MAterials
Machine Learning) packages, the [MatPES] (Materials Potential Energy Surface) dataset, and the [MatCalc] (Materials
Calculator).

## Documentation

The API documentation and tutorials are available at http://matcalc.ai.

## Outline

The main base class in MatCalc is `PropCalc` (property calculator). [All `PropCalc` subclasses](https://github.com/search?q=repo%3Amaterialsvirtuallab%2Fmatcalc%20%22(PropCalc)%22) should implement a
`calc(pymatgen.Structure | ase.Atoms | dict) -> dict` method that returns a dictionary of properties.

In general, `PropCalc` should be initialized with an ML model or [ASE] calculator, which is then used by either ASE,
LAMMPS or some other simulation code to perform calculations of properties. The `matcalc.PESCalculator` class
provides easy access to many foundation potentials (FPs) as well as an interface to MAML for custom MLIPs
such as MTP, NNP, GAP, etc.

# Basic Usage

MatCalc provides convenient methods to quickly compute properties, using a minimal amount of code. The following is
an example of a computation of the elastic constants of Si using the `TensorNet-MatPES-PBE-v2025.1-PES` universal MLIP.

```python
import matcalc as mtc
from pymatgen.ext.matproj import MPRester

mpr = MPRester()
si = mpr.get_structure_by_material_id("mp-149")
c = mtc.ElasticityCalc("TensorNet-MatPES-PBE-v2025.1-PES", relax_structure=True)
props = c.calc(si)
print(f"K_VRH = {props['bulk_modulus_vrh'] * 160.2176621} GPa")
```

The calculated `K_VRH` is about 102 GPa, in reasonably good agreement with the experimental and DFT values.

You can easily access a list of universal calculators (not comprehensive) using the UNIVERSAL_CALCULATORS enum.

```python
print(mtc.UNIVERSAL_CALCULATORS)
```

While we generally recommend users to specify exactly the model they would like to use, MatCalc provides useful
(case-insensitive) aliases to our recommended models for PBE and r2SCAN predictions. These can be loaded using:

```python
import matcalc as mtc
pbe_calculator = mtc.load_fp("pbe")
r2scan_calculator = mtc.load_fp("r2scan")
```

At the time of writing, these are the `TensorNet-MatPES-v2025.1` models for these functionals. However, these
recommendations may be updated as improved models become available.

MatCalc also supports trivial parallelization using joblib via the `calc_many` method.

```python
structures = [si] * 20

def serial_calc():
    return [c.calc(s) for s in structures]

def parallel_calc():
    # n_jobs = -1 uses all processors available.
    return list(c.calc_many(structures, n_jobs=-1))

%timeit -n 5 -r 1 serial_calc()
# Output is 8.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)

%timeit -n 5 -r 1 parallel_calc()
# Output is 2.08 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)
# This was run on 10 CPUs on a Mac.
```

MatCalc also supports chaining of `PropCalc`. Typically, you will start with a relaxation calc, followed by a series
of other calculators to get the properties you need. For example, the following snippet performs a relaxation,
followed by an energetics calculation and then an elasticity calculation. The final `results` contain all properties
computed by all steps. Note that the `relax_structure` should be set to False in later `PropCalc` to ensure that you
do not redo the relatively expensive relaxation.

```python
import matcalc as mtc
import numpy as np
calculator = mtc.load_fp("pbe")
relax_calc = mtc.RelaxCalc(
    calculator,
    optimizer="FIRE",
    relax_atoms=True,
    relax_cell=True,
)
energetics_calc = mtc.EnergeticsCalc(
    calculator,
    relax_structure=False  # Since we are chaining, we do not need to relax structure in later steps.
)
elast_calc = mtc.ElasticityCalc(
    calculator,
    fmax=0.1,
    norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
    shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
    use_equilibrium=True,
    relax_structure=False,  # Since we are chaining, we do not need to relax structure in later steps.
    relax_deformed_structures=True,
)
prop_calc = mtc.ChainedCalc([relax_calc, energetics_calc, elast_calc])
results = prop_calc.calc(structure)
```

Chaining can also be used with the `calc_many` method, with parallelization.

### CLI tool

A CLI tool provides a means to use FPs to obtain properties for any structure. Example usage:

```shell
matcalc calc -p ElasticityCalc -s Li2O.cif
```

## Benchmarking

MatCalc makes it easy to perform a large number of calculations rapidly. With the release of MatPES, we have released
the `MatCalc-Benchmark`.

For example, the following code can be used to run the ElasticityBenchmark on `TensorNet-MatPES-PBE-v2025.1-PES` FP.

```python
import matcalc as mtc

calculator = mtc.load_fp("TensorNet-MatPES-PBE-v2025.1-PES")
benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.05, relax_structure=True)
results = benchmark.run(calculator, "TensorNet-MatPES")
```

The entire run takes ~ 16mins when parallelized over 10 CPUs on a Mac.

You can even run entire suites of benchmarks on multiple models, as follows:

```python
import matcalc as mtc

tensornet = mtc.load_fp("TensorNet-MatPES-PBE-v2025.1-PES")
m3gnet = mtc.load_fp("M3GNet-MatPES-PBE-v2025.1-PES")

elasticity_benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.5, relax_structure=True)
phonon_benchmark = mtc.benchmark.PhononBenchmark(write_phonon=False)
suite = mtc.benchmark.BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
results = suite.run({"M3GNet": m3gnet, "TensorNet": tensornet})
results.to_csv("benchmark_results.csv")
```

These will usually take a long time to run. Running on HPC resources is recommended. Please set `n_samples` when
initializing the benchmark to limit the number of calculations to do some testing before running the full benchmark.

## Docker Images

Docker images with MatCalc and LAMMPS support are available at the [Materials Virtual Lab Docker Repository].

## Tutorials

Anubhav Jain (@computron) has created a nice [YouTube tutorial](https://youtu.be/57Elhe4IIhI?si=KbZh5s7HAyNGvmFT) on
how to use MatCalc to quickly obtain properties of materials.

## Citing

A manuscript on `matcalc` is currently in the works. In the meantime, please see [`citation.cff`](citation.cff) or the GitHub
sidebar for a BibTeX and APA citation.

[MAML]: https://materialsvirtuallab.github.io/maml/
[MatGL]: https://matgl.ai
[MatPES]: https://matpes.ai
[MatCalc]: https://matcalc.ai
[ASE]: https://wiki.fysik.dtu.dk/ase/
[Materials Virtual Lab Docker Repository]: https://hub.docker.com/u/materialsvirtuallab
