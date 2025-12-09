---
layout: default
title: Change Log
nav_order: 2
---

# Change Log

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
