---
layout: default
title: Change Log
nav_order: 2
---

# Change Log

## v0.7.0
1. **PR #271 New `DiffusionCalc` for diffusivity and ionic conductivity** by @rul048
   - Wraps pymatgen's `DiffusionAnalyzer`; runs MD via `MDCalc` or `LAMMPSMDCalc`, or analyses an existing `trajectory` in the input dict (e.g. after `MDCalc` in a `ChainedCalc`). Requires the new `diffusion` extra.
   - `MDCalc` accepts a velocity `seed`. `LAMMPSMDCalc` gains `seed`, `equilibration_frames`, `set_com_stationary` and `set_zero_rotation` (defaults preserve previous behavior).
   - `LAMMPSMDCalc` fixes: element order in `pair_coeff`/`dump_modify` follows the data-file type order, raises when `lmp` fails, averages energies over returned frames, returns the last frame as `final_structure` and all frames by default (`frames=None`), matching `MDCalc`.
2. **PR #272 New `DefectCalc` for neutral point-defect formation energies** by @rul048
   - `calc_defects` relaxes the bulk and generates vacancies, substitutions, antisites and interstitials (given coordinates, or Voronoi sites via the new `defects` extra); `calc` evaluates a single defect structure.
   - `NEBCalc.calc_endpoints` relaxes both endpoints at fixed cell.
   - Fixed `NEBCalc.calc_images` to interpolate along the shortest periodic displacement (`pbc=True`).
   - **Breaking:** the `NEBCalc` result key `force` previously held the final-minus-initial energy; that value is now `reaction_energy`, and `force` holds the NEB image forces (eV/Å).
3. **PR #273 Adhesion energy in `InterfaceCalc`** by @rul048
   - `adhesion_energy=True` returns `(E_film_slab + E_substrate_slab - E_interface) / A` (eV/Å²). New `calc_slab` and `calc_bulk` helpers; `relax_slab` (default True) relaxes slabs at fixed cell.
   - **Behavior change:** `interfacial_energy` is now the energy of a single film/substrate contact, `film_surface_energy + substrate_surface_energy - adhesion_energy`, and costs two extra fixed-cell slab relaxations. `calc` needs `film_slab`/`substrate_slab` or an `Interface` with `interface_label`; the composition-based atom-count fallback is removed. `interfacial_energy=False` skips bulk inputs entirely.
4. **PR #270 Unified foundation potential names and fixed model loading** by @rul048
   - Naming convention `<Architecture>-<Dataset>[-<Functional>]-<Version>[-<Variant>]`, with `<A>-ft-<B>` for fine-tuned models. Old names remain as case-insensitive aliases (e.g. `SevenNet-0` -> `SevenNet-MPtrj-0`, `MACE-MP-0-medium` -> `MACE-MPtrj-0a-medium`).
   - New models: `QET-MatPES-{PBE,r2SCAN}-2025.2`, `QET-MatQ`, `TensorNet-MatPES-{PBE,r2SCAN}-2025.2-m`, `M3GNet-MatPES-r2SCAN-2025.2`, `MACE-MH-{0,1}`, `SevenNet-Omni{,-i8,-i12}`, 26 GRACE models, and more.
   - Loader fixes for M3GNet MatPES 2025.2 (matgl >= 4.1.0), SevenNet `OAM-MF` modal, GRACE r6, ORB (orb-models >= 0.6 API), DeePMD built-in names; PET models now load via `upet`.
   - `load_universal` and `LAMMPSMDCalc` accept a local MatGL model directory; the CLI accepts any canonical name or alias and suggests close matches.
   - `invoke make-docs` generates `docs/models.md` from the registry.
   - **Breaking:** removed `M3GNet-MatPES-PBE-2025.1`, `PETMAD-1.0.0`, `DPA3-LAM-2025.3.14` (error names the replacement). Alias targets changed for `m3gnet`, `sevennet`, `grace`, `tensorpotential`, `orb`, `uma`, `fairchem`, `deepmd`. Extra `matcalc[petmad]` renamed to `matcalc[upet]`; ORB requires Python >= 3.12. Minimum versions bumped for matgl, mace-torch, sevenn, fairchem-core, deepmd-kit, orb-models.
5. **PR #269 MC and phonon rattle in `PhononCalc`; fixed random rattle** by @rul048
   - Each imaginary-mode correction retry now draws fresh displacements (new `seed`, default 42). New `rattle_stdev` (default 0.01 Å).
   - `rattle_method="mc"` and `rattle_method="phonon"` (optionally with `rattle_temperature`), adapted from hiphive. Default remains `"random"`.
6. **PR #268 `QHACalc` `write_energy_volume` option** by @rul048
   - Writes `e-v.dat` in `phonopy-qha` format so the QHA can be redone without new MLIP calculations.
7. **PR #284 / #285 `relax_calc_kwargs` now override all `RelaxCalc` arguments** by @shyuep
   - New `PropCalc._relaxer` builds every internal `RelaxCalc`; passing keys such as `fmax` or `relax_cell` in `relax_calc_kwargs` no longer raises `TypeError`. `DefectCalc` uses it directly, so `relax_calc_kwargs` also override `relax_cell`/`relax_atoms` there.
   - Foundation potential loader smoke tests moved to a path-triggered `test-fp-loaders.yml` workflow.
8. **PR #283 Tests run in parallel with pytest-xdist** by @shyuep
   - Per-test `tmp_path` isolation, controller-only cache setup, and fixes for order-dependent tests. CI suite time drops from ~290 s to ~60–70 s.
9. Test coverage improvements for CLI, MD, NEB, backend dispatch, surface, utils, interface, adsorption and stability modules (PRs #274–#282) by @shyuep.

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
- Added support for MatterSim, FAIRChem, PET-MAD, and DeePMD (@atulcthakur)

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
