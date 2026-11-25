# ruff: noqa: T201, INP001
"""Benchmark PhononCalc with the displaced supercells evaluated one at a time vs batched.

Run on a GPU node, e.g.::

    uv run python dev_scripts/bench_batched_phonon.py --device cuda --batch-sizes 1 4 8 16 32 64
    uv run --extra mace python dev_scripts/bench_batched_phonon.py --model MACE-MPA-0-medium --device cuda
    uv run --extra grace python dev_scripts/bench_batched_phonon.py --model GRACE-SMAX-OMat24-2L-medium

The structures are the matcalc test structures (Si, Li2O, LiFePO4); the supercell is sized by ``min_length``. Timings
cover the force evaluation and phonopy post-processing only (``relax_structure=False``), after one warm-up run (which
also absorbs GRACE's XLA compilation of the batched shape). ``--device`` applies to MatGL and MACE; GRACE
(TensorFlow) places itself on a visible GPU.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch
from pymatgen.util.testing import PymatgenTest

from matcalc import PhononCalc, load_fp


def main() -> None:
    """Time PhononCalc for each batch size and check that the force constants match the unbatched run."""
    parser = argparse.ArgumentParser(description="Benchmark PhononCalc with and without batched forces.")
    parser.add_argument("--model", default="TensorNet-MatPES-r2SCAN-2025.2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--structures", nargs="+", default=["Si", "Li2O", "LiFePO4"])
    parser.add_argument("--min-length", type=float, default=15.0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.model.upper().startswith("MACE"):
        calculator = load_fp(args.model, device=args.device)
    else:
        calculator = load_fp(args.model)
    if hasattr(calculator, "potential"):  # MatGL
        calculator.potential.to(device=args.device, dtype=torch.float32)

    def sync() -> None:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    print(f"model={args.model} device={args.device}")
    for name in args.structures:
        structure = PymatgenTest.get_structure(name)
        reference = None
        for batch_size in args.batch_sizes:
            calc = PhononCalc(
                calculator, min_length=args.min_length, relax_structure=False, write_phonon=False, batch_size=batch_size
            )
            calc.calc(structure)  # warm-up
            sync()
            start = time.perf_counter()
            result = calc.calc(structure)
            sync()
            elapsed = time.perf_counter() - start
            fc = result["phonon"].force_constants
            reference = fc if reference is None else reference
            n_disp = len(result["disp_supercells"])
            n_atoms = len(result["disp_supercells"][0])
            print(
                f"{name:8s} n_disp={n_disp:3d} n_atoms={n_atoms:4d} batch_size={batch_size:3d} "
                f"time={elapsed:7.2f}s max|dFC|={np.abs(fc - reference).max():.1e} eV/A^2"
            )


if __name__ == "__main__":
    main()
