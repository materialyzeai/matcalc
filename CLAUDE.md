# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

Dependency management uses `uv`. The `[ci]` extra installs everything needed for the full test suite (matgl, maml, matminer, mp-api, seekpath, jupyter/nbmake).

```bash
# Install for development (full test deps)
pip install -e '.[ci]'

# Run the test suite
pytest tests
pytest tests/test_elasticity.py                       # single file
pytest tests/test_elasticity.py::test_elasticity_calc # single test
pytest --cov=matcalc --cov-report=xml tests           # with coverage (CI invocation)

# Lint / format / typecheck (these are the CI gates)
ruff check src
ruff format src --check
mypy -p matcalc

# CLI entry point (installed as console script `matcalc`)
matcalc calc -p ElasticityCalc -s structure.cif
matcalc calc -p RelaxCalc -m TensorNet-MatPES-PBE-2025.2 -s structure.cif -o results.json
matcalc clear   # clears ~/.cache/matcalc benchmark data

# Release / docs automation (pyinvoke)
invoke make-docs        # builds Sphinx + tutorials into docs/
invoke update-changelog # generates changes.md entries from PR titles
invoke release          # creates GitHub release for current pyproject version
```

The `Test` workflow runs inside `docker.io/materialyzeai/lammps_gnnp` because LAMMPS-backed tests need a LAMMPS build with ML-GNNP and ML-SNAP. Locally, LAMMPS-dependent tests will be skipped or fail without that environment.

## Architecture

### The `PropCalc` contract

Everything in `src/matcalc/` is built around one base class in `_base.py`:

- `PropCalc.calc(structure) -> dict[str, Any]` is the single entry point. `structure` may be a `pymatgen.Structure`, an `ase.Atoms`, **or a dict** containing `final_structure` / `structure`. The dict form is what makes chaining work: each calculator's output dict is fed into the next calculator's `calc`. A subclass's first line is typically `result = super().calc(structure)` to normalize the input and pull out `final_structure`.
- `PropCalc.calc_many(structures, n_jobs=...)` parallelizes via `joblib.Parallel` and returns a generator. Pass `allow_errors=True` for large runs where you want failed structures to yield `None` instead of raising.
- `PropCalc.calculator` is an ASE `Calculator` instance. The setter accepts either an existing `Calculator` or a model-name string — strings are routed through `PESCalculator.load_universal()`. This is why every concrete calc accepts `calculator: str | Calculator` in its constructor.
- `ChainedCalc(prop_calcs)` runs calculators in sequence and accumulates their output dicts. The convention is to put a `RelaxCalc` first and then set `relax_structure=False` on downstream calcs to avoid re-relaxing.

### Foundation potential registry (`utils.py`)

`PESCalculator.load_universal(name)` (re-exported as `matcalc.load_fp` and `matcalc.load_up`) is the single dispatcher for loading any supported model. Two data structures drive it:

- `MODEL_REGISTRY` — canonical name → `{provider, ...kwargs}`. Naming convention is `<Architecture>-<Dataset>-<Optional Version>`, e.g. `TensorNet-MatPES-PBE-2025.2`.
- `MODEL_ALIASES` — short / legacy spellings (`pbe`, `r2scan`, `mace`, `chgnet`, …) that resolve to a canonical name. Lookups are case-insensitive.

To add a new model, add an entry to `MODEL_REGISTRY` and (if the provider is new) extend the loader branch in `PESCalculator.load_universal`. Provider stacks (matgl, mace-torch, sevenn, tensorpotential, orb-models, mattersim, fairchem-core, pet-mad, deepmd-kit) are imported lazily inside their branches so that missing optional deps don't break import of `matcalc`.

The `test-fp-loaders` matrix in `.github/workflows/test.yml` smoke-tests one canonical name per provider — keep it in sync when adding/removing canonical names for a covered provider.

### Backends (`backend/`)

Two simulation backends sit behind a thin dispatcher:

- `backend/_ase.py` (`run_ase`) — default; uses ASE optimizers/MD drivers.
- `backend/_lammps.py` (`run_lammps`) — used by `LAMMPSMDCalc` and friends.
- `backend/__init__.py::run_pes_calc` chooses based on `MATCALC_BACKEND` env var (read once into `config.SIMULATION_BACKEND`, default `"ASE"`).

`SimulationResult` (in `backend/_base.py`) is the common return shape — keep both backends returning structurally identical results so that downstream calcs don't branch on backend.

### Elemental references and units

- `src/matcalc/elemental_refs/*.json.gz` ships per-functional elemental references (MatPES-PBE, MatPES-r2SCAN, MP-PBE) used by `EnergeticsCalc` for formation/cohesive energies. `EnergeticsCalc` selects the reference set via the `functional` arg.
- `units.py` holds conversion factors. The most-used one is `eVA3ToGPa` for stress/elastic-tensor unit conversion — prefer it over hand-rolled constants.

### Benchmarks

`benchmark.py` defines `Benchmark`, `BenchmarkSuite`, and concrete `ElasticityBenchmark`, `PhononBenchmark`, etc. Benchmark datasets are pulled from the `materialyze/matcalc-bench` HuggingFace repo and cached in `~/.cache/matcalc` (configured in `config.py`). Use `n_samples` during development to avoid full-dataset runs.

## Conventions

- Single-underscore module names (`_elasticity.py`, `_phonon.py`, …) are implementation modules; the public API is what `src/matcalc/__init__.py` re-exports. When adding a new calculator, add the import there.
- All modules begin with `from __future__ import annotations` (enforced by ruff `isort.required-imports`). Ruff is run with `select = ["ALL"]`, so most lints are on; check `pyproject.toml` for the ignore list before disabling locally.
- Docstrings follow Google style. Type hints are required (mypy runs in CI; `no_implicit_optional = false`).
- `tests/conftest.py` provides session-scoped fixtures for common structures (`Si`, `Li2O`, `LiFePO4`, `SiO2`) and a `matpes_calculator` fixture. Reuse these instead of constructing structures inline. The autouse `setup_teardown` fixture cleans up files created during a test, so don't rely on artifacts persisting between tests.
- `tests/pes/` contains pre-trained classical-potential checkpoints (MTP, GAP, NNP, SNAP, ACE, NequIP, DPA3) used by `test_utils.py` to exercise the maml-backed loaders. These are real model files — don't move or rename them without updating the corresponding tests.
