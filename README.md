<h1 align="center">
  <img src="https://raw.githubusercontent.com/materialyzeai/matcalc/refs/heads/main/docs/assets/matcalc.png"
width="200" alt="MatCalc" style="vertical-align: middle;" /><br>
</h1>

[![Test](https://github.com/materialyzeai/matcalc/workflows/Test/badge.svg)](https://github.com/materialyzeai/matcalc/workflows/Test/badge.svg)
[![Lint](https://github.com/materialyzeai/matcalc/workflows/Lint/badge.svg)](https://github.com/materialyzeai/matcalc/workflows/Lint/badge.svg)
[![codecov](https://codecov.io/gh/materialyzeai/matcalc/branch/main/graph/badge.svg?token=OR7Z9WWRRC)](https://codecov.io/gh/materialyzeai/matcalc)
[![Requires Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/matcalc?logo=pypi&logoColor=white)](https://pypi.org/project/matcalc?logo=pypi&logoColor=white)
[![GitHub license](https://img.shields.io/github/license/materialyzeai/matcalc)](https://github.com/materialyzeai/matcalc/blob/main/LICENSE)

## Introduction

MatCalc is a Python library for calculating and benchmarking material properties from the potential energy surface
(PES). The PES can come from DFT or, more commonly, from machine learning interatomic potentials (MLIPs).

Calculating material properties often requires involved setups of various simulation codes. The
goal of MatCalc is to provide a simplified, consistent interface to access these properties with any
parameterization of the PES.

MatCalc is part of the MatML ecosystem, which includes [MatGL] (Materials Graph Library) and [MAML] (MAterials
Machine Learning), the [MatPES] (Materials Potential Energy Surface) dataset, and the [MatCalc] documentation site.

## Documentation

The API documentation and tutorials are available at https://matcalc.ai.

## Installation

```bash
pip install matcalc
```

For MatGL foundation potential support (TensorNet, M3GNet, CHGNet):

```bash
pip install matcalc[matgl]
```

For MAML classical potential support (MTP, GAP, NNP, SNAP):

```bash
pip install matcalc[maml]
```

For benchmarking support:

```bash
pip install matcalc[benchmark]
```

For phonon band paths via Seekpath:

```bash
pip install matcalc[phonon]
```

## Architecture

The main base class in MatCalc is `PropCalc` (property calculator). All `PropCalc` subclasses implement a
`calc(pymatgen.Structure | ase.Atoms | dict) -> dict` method that returns a dictionary of properties.

In general, `PropCalc` is initialized with an ML model or [ASE] calculator. The `matcalc.PESCalculator` class
provides convenient access to many foundation potentials (FPs) as well as an interface to MAML for classical
MLIPs such as MTP, NNP, GAP, SNAP, ACE, etc.

### Available Calculators

| Calculator | Description | Key Output Keys |
|---|---|---|
| `RelaxCalc` | Structural relaxation | `energy`, `forces`, `stress`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, `volume`, `final_structure` |
| `ElasticityCalc` | Elastic constants via strain-stress fitting | `elastic_tensor`, `bulk_modulus_vrh`, `shear_modulus_vrh`, `youngs_modulus`, `residuals_sum` |
| `EOSCalc` | Birch-Murnaghan equation of state | `eos`, `bulk_modulus_bm`, `r2_score_bm` |
| `PhononCalc` | Phonon band structure and thermal properties | `phonon`, `thermal_properties`, `frequencies`, `disp_supercells` |
| `Phonon3Calc` | Third-order force constants and thermal conductivity | `phonon3` (Phono3py object), `temperatures`, `thermal_conductivity` |
| `QHACalc` | Quasi-harmonic approximation | `gibbs_temperature`, `thermal_expansion`, `bulk_modulus_temperature` |
| `MDCalc` | Molecular dynamics simulation | `md_structures`, `md_energies` |
| `NEBCalc` | Nudged elastic band / minimum energy path | `mep`, `barrier_energy` |
| `EnergeticsCalc` | Formation and cohesive energies | `formation_energy_per_atom`, `cohesive_energy_per_atom` |
| `SurfaceCalc` | Surface energy calculation | `surface_energy`, `slab_energy`, `final_slab` |
| `AdsorptionCalc` | Adsorption energy on surfaces | `adsorption_energy` |
| `InterfaceCalc` | Coherent interface energy between two bulk structures | `interface_energy` |
| `LAMMPSMDCalc` | LAMMPS-based molecular dynamics | `md_structures`, `md_energies` |
| `ChainedCalc` | Chain multiple PropCalcs in sequence | Combined outputs of all constituent calculators |

### Supported Foundation Potentials

`PESCalculator.load_universal()` — aliased as `matcalc.load_fp()` and `matcalc.load_up()` — supports these models out of the box:

| Canonical name | Common aliases | Package |
|---|---|---|
| `TensorNet-MatPES-PBE-2025.2` | `tensornet`, `pbe` | matgl |
| `TensorNet-MatPES-r2SCAN-2025.2` | `r2scan` | matgl |
| `M3GNet-MatPES-PBE-2025.2` | `m3gnet` | matgl |
| `M3GNet-MatPES-r2SCAN-2025.2` | — | matgl |
| `QET-MatPES-{PBE,r2SCAN}-2025.2` | `qet` | matgl |
| `QET-MatQ` | — | matgl |
| `CHGNet-MatPES-PBE-2026.9-1M` | `chgnet` | matgl |
| `CHGNet-MatPES-r2SCAN-2026.9-1M` | — | matgl |
| `CHGNet-MatPES-{PBE,r2SCAN}-2025.2.10` | — | matgl |
| `MACE-MPtrj-0a-{small,medium,large}` | — | mace-torch |
| `MACE-MPtrj-0b-{small,medium}` | — | mace-torch |
| `MACE-MPtrj-0b2-{small,medium,large}` | — | mace-torch |
| `MACE-MPtrj-0b3-medium` | — | mace-torch |
| `MACE-MPA-0-medium` | `mace` | mace-torch |
| `MACE-OMat24-0-{small,medium}` | — | mace-torch |
| `MACE-MatPES-{PBE,r2SCAN}-0` | — | mace-torch |
| `SevenNet-MPtrj-0` | `sevennet` | sevenn |
| `SevenNet-MPtrj-l3i5` | — | sevenn |
| `SevenNet-OAM-MF` | — | sevenn |
| `SevenNet-OMat24` | — | sevenn |
| `GRACE-SMAX-OMat24-2L-medium` | `grace`, `tensorpotential` | tensorpotential |
| `GRACE-SMAX-OMat24-{1L-large,2L-large}`, `GRACE-SMAX-{1L-large,2L-medium,2L-large}` | — | tensorpotential |
| `GRACE-OAM-{1L,2L}-{small,medium,large}`, `GRACE-OAM-3L-large` | — | tensorpotential |
| `GRACE-OMat24-{1L,2L}-{medium,large}-{base,ft-E}`, `GRACE-OMat24-2L-small`, `GRACE-OMat24-3L-large` | — | tensorpotential |
| `GRACE-MPtrj-{1L-r6,2L-r5,2L-r6}` | — | tensorpotential |
| `ORB-MPA-v2` | `orb` | orb-models |
| `ORB-MPA-v2-D3`, `ORB-MPA-v2-D3-{sm,xs}` | — | orb-models |
| `MatterSim-v1.0.0-1M` | `mattersim` | mattersim |
| `MatterSim-v1.0.0-5M` | — | mattersim |
| `UMA-S-1.2` | `uma`, `fairchem` | fairchem-core |
| `UMA-S-{1.1,1.2.1}`, `UMA-M-1.1` | — | fairchem-core |
| `PET-MAD-1.0.2` | `petmad` | upet |
| `DPA2-OpenLAM-2.4-7M` | — | deepmd-kit |
| `DPA3-OpenLAM-{3.1-3M,3.2-5M,3.3-1M}` | `deepmd`, `dpa` | deepmd-kit |
| `DPA4-OMat24-2026.8.5-{Nano,Mini,Neo,Air,Plus}` | — | deepmd-kit |

Names follow `<Architecture>-<Dataset>[-<Functional>]-<Version>[-<Variant>]`; braces list variants.
Aliases are case-insensitive. All pretrained MatGL PES models are auto-discovered if MatGL is installed.

The 1M CHGNet models were trained with the CHGNet line-graph fix from [matgl#843](https://github.com/materialyzeai/matgl/pull/843). Until that fix is in a matgl release, install the development version (`pip install git+https://github.com/materialyzeai/matgl.git`) to get exact results for cells with a lattice vector shorter than 3 Å (the CHGNet three-body cutoff).

The table is not exhaustive. Other models from these packages can be loaded with a provider override,
e.g. `load_fp("grace", model="GRACE-2L-SMAX-OMAT-large-fp64")`, or by passing any ASE calculator directly,
e.g. a fairchem 1.x model (`fairchem-core==1.10`):

```python
import matcalc as mtc
from fairchem.core import OCPCalculator

calc = OCPCalculator(model_name="eSEN-30M-OAM", local_cache="fairchem_checkpoints", cpu=False)
results = mtc.RelaxCalc(calc, relax_cell=True).calc(structure)
```

## Basic Usage

MatCalc provides convenient methods to quickly compute properties with minimal code. The following example
computes the elastic constants of Si using the `TensorNet-MatPES-PBE-2025.2` universal MLIP.

```python
import matcalc as mtc
from pymatgen.ext.matproj import MPRester

mpr = MPRester()
si = mpr.get_structure_by_material_id("mp-149")
c = mtc.ElasticityCalc("TensorNet-MatPES-PBE-2025.2", relax_structure=True, units_GPa=True)
props = c.calc(si)
print(f"K_VRH = {props['bulk_modulus_vrh']} GPa")
```

The calculated `K_VRH` is about 102 GPa, in reasonably good agreement with the experimental and DFT values.

By default, `ElasticityCalc` returns the elastic tensor and moduli in pymatgen's native units of
eV/Å³. Pass `units_GPa=True` (as above) to have `elastic_tensor`, `bulk_modulus_vrh`,
`shear_modulus_vrh`, `youngs_modulus`, and `residuals_sum` returned directly in GPa, avoiding manual
unit conversion. The returned dict also carries a `_units` entry mapping each numeric output to its
unit string.

You can list all supported universal calculators using the `UNIVERSAL_CALCULATORS` enum:

```python
print(mtc.UNIVERSAL_CALCULATORS)
```

MatCalc provides case-insensitive aliases for the recommended PBE and r²SCAN models:

```python
import matcalc as mtc
pbe_calculator = mtc.load_fp("pbe")
r2scan_calculator = mtc.load_fp("r2scan")
```

These currently resolve to the `TensorNet-MatPES-*-2025.2` models, but may be updated as better models
become available.

`matcalc.load_up` is the same as `matcalc.load_fp` (historical alias).

### Parallelization

MatCalc supports trivial parallelization via joblib through the `calc_many` method:

```python
structures = [si] * 20

def serial_calc():
    return [c.calc(s) for s in structures]

def parallel_calc():
    # n_jobs = -1 uses all available processors.
    return list(c.calc_many(structures, n_jobs=-1))

%timeit -n 5 -r 1 serial_calc()
# Output: 8.7 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)

%timeit -n 5 -r 1 parallel_calc()
# Output: 2.08 s ± 0 ns per loop (mean ± std. dev. of 1 run, 5 loops each)
# This was run on 10 CPUs on a Mac.
```

### Chaining Calculators

`ChainedCalc` runs a sequence of `PropCalc` instances on a structure, accumulating all output properties.
Typically, you start with a `RelaxCalc` followed by property calculators. Set `relax_structure=False` in
downstream calculators to avoid redundant relaxations.

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
    relax_structure=False  # Skip re-relaxation since we already relaxed above.
)
elast_calc = mtc.ElasticityCalc(
    calculator,
    fmax=0.1,
    norm_strains=list(np.linspace(-0.004, 0.004, num=4)),
    shear_strains=list(np.linspace(-0.004, 0.004, num=4)),
    use_equilibrium=True,
    relax_structure=False,  # Skip re-relaxation since we already relaxed above.
    relax_deformed_structures=True,
    units_GPa=True,  # Report moduli in GPa rather than eV/A^3.
)
prop_calc = mtc.ChainedCalc([relax_calc, energetics_calc, elast_calc])
results = prop_calc.calc(structure)
```

`ChainedCalc` also works with `calc_many` for parallel execution over many structures.

### CLI Tool

A command-line interface allows computing properties for any structure file:

```shell
# Compute elastic constants for a CIF file using the default TensorNet model
matcalc calc -p ElasticityCalc -s Li2O.cif

# Use a specific model and write output to a JSON file
matcalc calc -p RelaxCalc -m TensorNet-MatPES-PBE-2025.2 -s structure.cif -o results.json

# Clear the local benchmark data cache
matcalc clear
```

Any format supported by pymatgen's `Structure.from_file()` method is accepted for input structures.
Available property calculators can be checked with `matcalc calc -h`.

## Benchmarking

MatCalc includes a benchmarking framework released alongside [MatPES].

```python
import matcalc as mtc

calculator = mtc.load_fp("TensorNet-MatPES-PBE-2025.2")
benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.05, relax_structure=True)
results = benchmark.run(calculator, "TensorNet-MatPES")
```

The entire run takes about 16 minutes when parallelized over 10 CPUs on a Mac.

To run multiple benchmarks on multiple models:

```python
import matcalc as mtc

tensornet = mtc.load_fp("TensorNet-MatPES-PBE-2025.2")
m3gnet = mtc.load_fp("M3GNet-MatPES-PBE-2025.2")

elasticity_benchmark = mtc.benchmark.ElasticityBenchmark(fmax=0.5, relax_structure=True)
phonon_benchmark = mtc.benchmark.PhononBenchmark(write_phonon=False)
suite = mtc.benchmark.BenchmarkSuite(benchmarks=[elasticity_benchmark, phonon_benchmark])
results = suite.run({"M3GNet": m3gnet, "TensorNet": tensornet})
results.to_csv("benchmark_results.csv")
```

Full benchmark runs are computationally intensive. Set `n_samples` when initializing a benchmark to test on a
subset before running the full suite. HPC resources are recommended for full benchmark runs.

## Docker Images

Docker images with MatCalc and LAMMPS support are available at the [Materialyze AI Docker Repository].

## Tutorials

Anubhav Jain (@computron) has created a [YouTube tutorial](https://youtu.be/57Elhe4IIhI?si=KbZh5s7HAyNGvmFT) on
using MatCalc to quickly obtain material properties.

## FAQs

### Which MLIP or foundation potential should I use?

Think of a model as **architecture + training data**, and pick the dataset first — it usually matters more than the
architecture. We currently recommend three:

- **MatPES** (PBE or r2SCAN) for solids
- **OMat24** for solids where rattled-structure coverage matters (e.g. phonons)
- **OMol25** for molecules

Most other public models are trained on **MPtrj**, which is noisy and outdated; we don't recommend it. The [MatPES]
website provides head-to-head benchmarks between MatPES and OMat24 to help you choose.

**MatPES** uses the latest pseudopotentials with tight energy and force convergence, and is also available at the
**r2SCAN** level. MatPES-trained models tend to be the most stable and give excellent property predictions across the
board. The main caveat is the absence of a Hubbard U, so PBE-trained MatPES models can be off for oxidation-state
energies, voltages, and similar quantities — the r2SCAN variant largely addresses this.

**OMat24** suffers from PBE / PBE+U discontinuities and looser convergence criteria, but its rattled structures tend to
yield slightly better phonons and phonon-derived properties.

Once the dataset is fixed, there is little to separate the architectures. We default to **TensorNet** for its parameter
efficiency and speed; **GRACE** and **MACE** are also excellent choices. TensorNet and MACE have MatPES-trained checkpoints
available via `matcalc.load_fp`.

There is no substitute for benchmarking on the property you actually care about — MatCalc is built to make that easy.

## Citing

A manuscript on `MatCalc` is currently in the works. In the meantime, please see [`citation.cff`](citation.cff) or the GitHub
sidebar for a BibTeX and APA citation.

[MAML]: https://materialyzeai.github.io/maml/
[MatGL]: https://matgl.ai
[MatPES]: https://matpes.ai
[MatCalc]: https://matcalc.ai
[ASE]: https://wiki.fysik.dtu.dk/ase/
[Materialyze AI Docker Repository]: https://hub.docker.com/orgs/materialyzeai/repositories
