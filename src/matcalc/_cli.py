"""Command line interface to matcalc."""

from __future__ import annotations

import argparse
import logging
import pprint
import typing

from monty.json import jsanitize
from monty.serialization import dumpfn
from pymatgen.core import Structure

import matcalc as mtc

if typing.TYPE_CHECKING:
    from typing import Any


# Explicit map of PropCalc subclasses exposed via the CLI. Dict-based dispatch
# avoids the previous ``mtc.__dict__[args.property]`` lookup, which silently
# broke whenever a class was renamed or removed.
# NEBCalc is intentionally excluded because it does not accept a single
# Structure and returns a non-dict result (MEP namedtuple-like dataclass).
# Typed as ``Any`` because each subclass adds its own ``__init__`` arguments
# (mypy infers the base ``PropCalc.__init__`` signature otherwise).
CLI_CALCS: dict[str, type] = {
    name: getattr(mtc, name)
    for name in (
        "RelaxCalc",
        "ElasticityCalc",
        "EOSCalc",
        "PhononCalc",
        "Phonon3Calc",
        "QHACalc",
        "MDCalc",
        "EnergeticsCalc",
        "SurfaceCalc",
        "GBCalc",
        "InterfaceCalc",
        "AdsorptionCalc",
        "LAMMPSMDCalc",
    )
    if hasattr(mtc, name)
}


def calculate_property(args: Any) -> None:
    """Run the selected PropCalc on input structure files (from parsed CLI args)."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    calculator = mtc.load_fp(args.model)
    calc_cls = CLI_CALCS[args.property]
    mod = calc_cls(calculator)
    results = []
    for f in args.structure:
        s = Structure.from_file(f)
        results.append(mod.calc(s))
    if args.outfile:
        if "json" not in args.outfile:
            dumpfn(jsanitize(results), args.outfile)
        else:
            dumpfn(results, args.outfile)
    else:
        pprint.pprint(results)  # noqa:T203


def clear_cache(args: Any) -> None:
    """Clear the MatCalc benchmark cache (from parsed CLI args)."""
    mtc.clear_cache(confirm=args.yes)


def main() -> None:
    """Handle main."""
    parser = argparse.ArgumentParser(
        description="""A CLI interface for rapid calculations of materials properties with matcalc. Type
        "matcalc sub-command -h".""",
        epilog="""Author: MatCalc Development Team""",
    )

    subparsers = parser.add_subparsers()

    p_calc = subparsers.add_parser("calc", help="Calculate properties using universal calculators.")

    p_calc.add_argument(
        "-s",
        "--structure",
        dest="structure",
        nargs="+",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_calc.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        choices=mtc.UNIVERSAL_CALCULATOR_NAMES,
        default="TensorNet-MatPES-PBE-2025.2",
        help="Universal MLIP to use.",
    )

    p_calc.add_argument(
        "-p",
        "--property",
        dest="property",
        type=str,
        choices=sorted(CLI_CALCS),
        default="RelaxCalc",
        help="PropCalc to use. Defaults to RelaxCalc.",
    )

    p_calc.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        nargs="?",
        help="Output file in json or yaml. Defaults to stdout.",
    )

    p_calc.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Enable DEBUG-level logging from matcalc.",
    )

    p_calc.set_defaults(func=calculate_property)

    p_clear = subparsers.add_parser("clear", help="Clear cache.")

    p_clear.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip confirmation.",
    )

    p_clear.set_defaults(func=clear_cache)

    args = parser.parse_args()

    return args.func(args)
