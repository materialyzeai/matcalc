---
layout: default
title: Change Log
nav_order: 2
---

# Change Log

## v0.6.0
1. **PR #224 New `OrderCalc` for Monte Carlo ordering of disordered structures** by @shyuep
   - Builds a random ordering commensurate with site occupancies (partial occupancy < 1 treated as vacancies), then runs Metropolis MC swaps on disordered sites using the PES calculator.
   - Supports early stopping (`patience`), multiple parallel chains (`n_init`, `n_jobs`), and returning the `n_lowest` distinct orderings (`lowest_structures` / `lowest_energies`).
   - `OrderCalc` now handles `Element` and `DummySpecies` sites in addition to `Species` (@rul048).
2. **PR #267 Default CHGNet switched to the 1M MatPES models** by @YiqingChen524
   - Registered `CHGNet-MatPES-PBE-1M-2026.9` and `CHGNet-MatPES-r2SCAN-1M-2026.9`; the `chgnet` alias and LAMMPS alias now point to them. The 2.7M `2025.2.10` names still resolve.
3. **PR #263 Added `UMA-S-1.2.1` to the model registry** by @Andrew-S-Rosen
4. **PR #233 Switch from deprecated phonopy dictionaries to attributes; `phono3py` now optional** by @Andrew-S-Rosen, @rul048
   - Install via the `phonon3` extra (`pip install matcalc[phonon3]`). Minimum versions bumped to `phonopy>=4.2.0` and `phono3py>=4.2.0`.
   - `PhononCalc` builds the `thermal_properties` dict directly instead of the deprecated `get_thermal_properties_dict()`.
5. **PR #230 `PhononCalc` / `QHACalc` use `primitive_matrix="P"` instead of `"auto"`** by @Andrew-S-Rosen
6. **PR #227 Reverted the QHA volume-drift skip**: a volume change during fixed-cell relaxation raises again rather than silently dropping the scale factor. By @Andrew-S-Rosen
7. **PR #234 Fixed `QHACalc` units**: `gibbs_free_energies` is in eV, not kJ/mol (docstring and `_units`). By @Andrew-S-Rosen
8. **PR #242 Renamed `PropCalc._prerelax` to `_check_and_prelax`** by @shyuep
9. Docs: `ElasticityCalc` examples use `units_GPa=True` (PR #243); added citation info for Andrew Rosen (PR #266).

## v0.5.1
1. CI hygiene: drop `dgl` install pin from all GitHub Actions workflows (matgl now resolves its own backend); fix `release.yml` permissions/`secrets: inherit` so the reusable `test.yml` call satisfies the subset rule and PyPI trusted publishing works.

## v0.5.0
1. **`RelaxCalc` convergence checks.**
   - `RelaxCalc.calc` surfaces `is_converged` and `max_force` in its result dict.
   - `PropCalc._prerelax` raises `RuntimeError` when the pre-relaxation fails to converge, instead of silently feeding an unrelaxed structure into downstream property calcs.
   - `RelaxCalc` now copies the structure before perturbing, eliminating a caller-mutation hazard.
2. **PR #218 Extract `PropCalc._prerelax` helper; fold relax boilerplate** by @shyuep
   - Centralises the `if self.relax_structure: relaxer = RelaxCalc(...); result |= relaxer.calc(structure); structure = result['final_structure']` pattern duplicated across `_phonon3`, `_eos`, `_qha`, `_md`, `_stability`, and `_elasticity`.
3. **PR #210 / #85 Fix `youngs_modulus` and add `_units` metadata across all `PropCalc`s** by @shyuep
   - Self-consistent `youngs_modulus` computation; every `PropCalc` result now carries a `_units` dict, with assertions in the test suite.
4. **`Phonon3Calc` adapted to the new phono3py conductivity API.**
5. **Foundation potential model naming unified.** UMA model names updated to match FAIRChem upstream; CHGNet alias typo fixed; small registry cleanups.

## v0.4.8
1. Updated the Hugging Face organization to `materialyze` (lower case).

## v0.4.7
1. **`ElasticityCalc` can now return moduli in GPa.**
   - Added a new `units_GPa` keyword argument (default `False`). When set to `True`, the returned `elastic_tensor`, `bulk_modulus_vrh`, `shear_modulus_vrh`, and `residuals_sum` are converted from pymatgen's native eV/A^3 units to GPa.
2. **Bug fix in `EOSCalc`.**
   - Fixed a variable shadowing issue where the loop-local relaxation result was overwriting the outer merged `result` dict, causing the merged relaxation/structure fields returned by `EOSCalc.calc` to come from a strained calculation rather than the originally relaxed input structure.

## v0.4.6
1. **PR #199 Add `GBCalc` Class for Grain Boundary Calculation** (originally PR #111)
   - Author: @rul048
   - Description: Adds `GBCalc` for generating grain boundary structures and computing grain boundary energies, complementing the existing `SurfaceCalc` and `InterfaceCalc` classes.
2. **PR #203 Multiple pressure support in `QHACalc`**
   - Author: @Andrew-S-Rosen
   - Description: Adds support for evaluating multiple pressures in the quasi-harmonic approximation workflow.
3. **PR #174, #176, #163 Phonon and QHA refactor**
   - Author: @Andrew-S-Rosen
   - Description: Significant refactor of `PhononCalc` and `QHACalc` for clearer logic, more consistent relaxation settings, better error handling, and improved typing.
   - Description: Adds detection and improved handling of imaginary phonon modes.
   - Description: Fixes `PhononCalc` handling of `Atoms` inputs and supercell generation, and exposes a `min_length` kwarg for supercell sizing.
   - Description: Allows users to toggle `symprec` in `Phonopy` calls and sets `primitive_matrix="auto"` by default.
4. **PR #158 Consistent relaxation settings in `QHACalc`**
   - Author: @Andrew-S-Rosen
   - Description: Relaxes atomic positions in the QHA workflow and ensures consistent settings between relaxations.
5. **PR #179 Allow fixing symmetry or specific atoms during relaxation**
   - Author: @rul048
   - Description: Adds support in `RelaxCalc` for fixing symmetry or specifying particular atoms to be held fixed during relaxation.
6. **PR #161 / #206 Better handling when relaxations hit max steps**
    - Author: @Andrew-S-Rosen
    - Description: Logs (instead of warning) when a relaxation reaches its maximum step count, making the behavior easier to track in long-running workflows.
7. **PR #189 Remove upper triangular cell call in NVT ensemble**
    - Author: @Andrew-S-Rosen
    - Description: Avoids unnecessary cell transformation in `MDCalc` NVT runs.
8. **PR #136 Default NEB method to `improvedtangent`**
    - Author: @rul048
    - Description: Changes the default NEB method to `improvedtangent` for more reliable minimum energy paths.
9. Replaced the custom `TrajectoryObserver` with ASE's native trajectory observer.

## v0.4.5
1. **PR #117 Add `InterfaceCalc` Class for Interface Structure and Energy Calculation**
   - Author: @so2koo
   - Description: Introduces `InterfaceCalc` for calculating interface structures and energies, similar to `SurfaceCalc`. Utilizes `pymatgen CoherentInterfaceBuilder` to generate interface structures from two different bulk structures. Employs `RelaxCalc` for structure relaxation and determines the lowest interface energy.
   - Todos: Further work may be needed if this is a work in progress.
2. **PR #118 Add `AdsorptionCalc` Class for Adsorption Energy Calculation**
   - Author: @mkphuthi
   - Description: Adds `AdsorptionCalc` for calculating adsorption energy of single molecules on surfaces. Capabilities include generating slabs with `SlabGenerator`, placing adsorbates with `AdsorbateSitefinder`, and directly supplying an adslab. Features include calculation of adsorption energies, support for customizable relaxation via `RelaxCalc` and ASE optimizers, extensive documentation, and type hints.
   - Todos: Some unrelated tests need to be fixed.

## v0.4.4
1. More data-efficient MEP class for storing NEB results.

## v0.4.3
1. **PR #102 fix:** add `required-environments` to enable Linux install by @MorrowChem
2. **PR #97:** Make it possible to set COM momentum and/or rotation to zero in `MDCalc` by @Andrew-S-Rosen
   - Feature: Allows setting the center-of-mass momentum and angular momenta to zero in `MDCalc` to prevent drift.
3. **PR #96:** Allow for `MDCalc` to be run with `relax_cell=True` during the initial relaxation by @Andrew-S-Rosen
   - Feature: Enables unit cell relaxation before an MD run.
4. **PR #89:** Add support for `MTKNPT` by @Andrew-S-Rosen
   - Feature: Added support for `MTKNPT`, the recommended method for NPT simulations in ASE.
5. **PR #90:** Add a warning if using `NPT` by @Andrew-S-Rosen
    - Warning: Raises `UserWarning` when `NPT` class is used due to stability concerns.
6. **PR #88:** Add IsotropicMTKNPT to `_md.py` by @Andrew-S-Rosen
    - Feature: Introduced `IsotropicMTKNPT` support, expanding supported `ensemble` options.
7. Progress bars when running benchmarks.

## v0.4.2
- Bug fix for surface calculations (@computron).
- Update OCPCalculator with the newer FairChemCalculator (@atulcthakur)

## v0.4.1
- Bug fix for bad trajectory snapshotting in MDCalc.
- Beta LAMMPSMDCalc.

## v0.4.0
- All PropCalcs now support ASE Atoms as inputs, as well as pymatgen Structures.
- Added MDCalc class for molecular dynamics simulations. (@rul048)
- Minor updates to EquilibriumBenchmark to make it easier to reproduce matpes.ai/benchmarks results.  (@rul048)

## v0.3.3
- SurfaceCalc class for computing surface energies. (@atulcthakur)
- NEBCalc now returns MEP information. (@drakeyu)
- All PropCalcs and Benchmarks now support using a string as a calculator input, which is automatically interpreted as
  a universal calculator where possible. This greatly simplifies the use of PropCalc.
- PESCalculator.load_universal now is lru_cached so that the same model does not get loaded multiple times.

## v0.3.2
- Added Phonon3Calc for calculation of phonon-phonon interactions and thermal conductivity using Phono3py. (@rul48)

## v0.3.1
- All PropCalc implementations are now private and should be imported from the base package.
- Added support for Mattersim, Fairchem-Core, PET-MAD, and DeepMD (@atulcthakur)

## v0.2.2
- Added ChainedCalc helper class to performed chaining of PropCalcs more easily.
- Added matcalc.load_fp alias for easier loading of universal MLIPs.

## v0.2.0
- Major new feature: Most PropCalc now supports chaining. This allows someone to obtain multiple properties at once.
- SofteningBenchmark added. (@bowen-bd)
- Major expansion in the number of supported calculators, including Orb and GRACE models. (@atulcthakur)

## v0.1.2
- Emergency bug fix for bad default perturb_distance parameter in Relaxation.

## v0.1.1
- Provide model aliases "PBE" and "R2SCAN" which defaults to the TensorNet MatPES models.
- New CLI tools to quickly use prop calculators to compute properties from structure files.

## v0.1.0
- Added support for ORB and GRACE universal calculators (@atulcthakur)
- Option to perturb structure before relaxation (@rul048)
- Improved handling of stress units (@rul048)
- Option to relax strained structures in ElasticityCalc (@lbluque)

## v0.0.6
- Checkpointing and better handling of benchmarking.
- Most PropCalc can now be imported from the root level, e.g., `from matcalc import ElasticityCalc` instead of the more
  verbose `from matcalc.elasticity import ElasticityCalc`.

## v0.0.5

- Initial release of benchmarking tools with Elasticity and Phonon benchmark data.

## v0.0.2

- Minor updates to returned dicts.

## v0.0.1

- First release with all major components.
