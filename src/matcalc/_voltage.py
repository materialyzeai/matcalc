"""Calculator for intercalation voltage profiles from Monte Carlo sampling."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from monty.serialization import loadfn
from pymatgen.apps.battery.insertion_battery import InsertionElectrode
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

from ._base import PropCalc
from ._intercalation import IntercalationCalc, _check_removal_range, removal_counts
from ._relaxation import RelaxCalc
from .utils import to_pmg_structure

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator

logger = logging.getLogger(__name__)

ELEMENTAL_REFS_DIR = Path(__file__).parent / "elemental_refs"

# Relaxation defaults shared by the host prelaxation, the Monte Carlo proposals (MCCalc uses the same
# values) and the working-ion reference, so that all energies entering a voltage are consistent.
_RELAX_DEFAULTS: dict[str, Any] = {"optimizer": "FIRE", "fmax": 0.02, "max_steps": 200}


class VoltageCalc(PropCalc):
    """Intercalation voltage profile from Monte Carlo sampling of working-ion arrangements.

    The (optionally relaxed and supercelled) host is delithiated level by level with
    :class:`IntercalationCalc`, which samples the arrangement of the remaining working ions at each
    ion count. The lowest energy found at each level, together with a working-ion metal reference
    relaxed with the same calculator, gives:

    - raw step voltages between consecutive ion counts, which line up across models run with the
      same seed and are intended for benchmarking, and
    - the convex-hull voltage profile, average voltage, capacity and energy density from pymatgen's
      :class:`~pymatgen.apps.battery.insertion_battery.InsertionElectrode`.

    The working-ion valence follows pymatgen's convention (the element's highest oxidation state,
    e.g. 1 for Li and Na, 2 for Mg), so raw and hull voltages are directly comparable.

    Note:
        Like :class:`IntercalationCalc`, this is slow (a relaxation per Monte Carlo proposal) and is
        intended for benchmarking, e.g. quantifying the error in voltages across models by sampling
        from the same seed, not for production voltage-profile calculations.
    """

    def __init__(
        self,
        calculator: Calculator | str,
        *,
        species: str = "Li",
        removal_range: tuple[float, float] = (0.0, 1.0),
        supercell: Sequence[int] | None = None,
        nsteps: int = 100,
        temperature: float = 300,
        algorithm: Literal["independent", "markov"] = "independent",
        seed: int | None = None,
        relax: bool = True,
        relax_structure: bool = True,
        ion_reference: Literal["MatPES-PBE", "MatPES-r2SCAN", "MP-PBE"] | float | Structure = "MatPES-PBE",
        relax_calc_kwargs: dict[str, Any] | None = None,
        intercalation_calc_kwargs: dict[str, Any] | None = None,
        trajfile: str = "voltage_mc.traj",
        store_intercalation: bool = True,
        write_voltage_profile: bool | str | os.PathLike = False,
    ) -> None:
        """Initialize the VoltageCalc.

        Args:
            calculator: ASE calculator or universal model name string.
            species: Working-ion element symbol. Default is ``"Li"``.
            removal_range: Inclusive (start, stop) fractions of the working-ion sites to remove, e.g.
                ``(0.0, 0.5)`` for a material that only delithiates to 50%. Every whole number of ions in
                the range is sampled, one ion at a time, so the supercell sets the resolution; bounds are
                rounded inward with a warning. At least two ion counts are required. Default is
                ``(0.0, 1.0)``.
            supercell: Supercell scaling applied to the (relaxed) host before sampling. Default is None.
            nsteps: Monte Carlo steps per ion count. Levels with a single configuration (the end
                points) are scored once. Default is 100.
            temperature: Monte Carlo temperature in Kelvin. Default is 300.
            algorithm: ``"independent"`` or ``"markov"`` sampling, see :class:`IntercalationCalc`.
                Default is ``"independent"``.
            seed: Seed for reproducible sampling. Use the same seed to compare models. Default is None.
            relax: Relax every Monte Carlo proposal, and the working-ion reference. If False, single
                points are used for both. Default is True.
            relax_structure: Relax the input host before sampling. Default is True.
            ion_reference: Working-ion metal reference. A built-in elemental reference set
                (``"MatPES-PBE"``, ``"MatPES-r2SCAN"`` or ``"MP-PBE"``) or a ``Structure``, which is
                relaxed (or single-pointed if ``relax`` is False) with ``calculator``; or a float,
                used directly as the energy per atom in eV. Default is ``"MatPES-PBE"``.
            relax_calc_kwargs: Extra ``RelaxCalc`` arguments for every relaxation (host, proposals and
                reference); they override the defaults ``optimizer="FIRE"``, ``fmax=0.02``,
                ``max_steps=200``. Default is None.
            intercalation_calc_kwargs: Extra :class:`IntercalationCalc` arguments (e.g.
                ``disordered_structure``, ``swap_size``, ``save_freq``); they override the values set
                from this calculator's arguments. Default is None.
            trajfile: Base Monte Carlo trajectory filename; a ``_k{k}`` suffix is added per level.
                Default is ``"voltage_mc.traj"``.
            store_intercalation: Store the raw :class:`IntercalationCalc` results under the
                ``"intercalation"`` key. Default is True.
            write_voltage_profile: Write the per-level energies and step voltages as CSV. True writes
                ``voltage_profile.csv``, a path selects the file, False disables writing.
                Default is False.
        """
        self.calculator = calculator  # type: ignore[assignment]
        self.species = species
        self.valence = abs(max(Element(species).oxidation_states, default=0))
        if self.valence == 0:
            raise ValueError(f"{species} has no positive oxidation state and cannot be a working ion")
        _check_removal_range(removal_range)
        self.removal_range = removal_range
        self.supercell = supercell
        self.nsteps = nsteps
        self.temperature = temperature
        self.algorithm = algorithm
        self.seed = seed
        self.relax = relax
        self.relax_structure = relax_structure
        self.ion_reference = ion_reference
        self.relax_calc_kwargs = relax_calc_kwargs or {}
        self.intercalation_calc_kwargs = intercalation_calc_kwargs or {}
        self.trajfile = trajfile
        self.store_intercalation = store_intercalation
        self.write_voltage_profile: str | os.PathLike | None = (
            "voltage_profile.csv" if write_voltage_profile is True else (write_voltage_profile or None)  # type: ignore[assignment]
        )

    def calc(self, structure: Structure | Atoms | dict[str, Any]) -> dict[str, Any]:
        """Compute the voltage profile of a fully intercalated host.

        Args:
            structure: Fully intercalated host structure, ASE atoms, or a dict with structure keys.

        Returns:
            Dict with, per ion count sorted by increasing ``x`` (fraction of working-ion sites
            occupied, within ``x_range``): ``x``, ``n_removed``, ``energies`` (final accepted, eV), ``min_energies``
            (lowest found, eV), ``acceptance_ratios``, ``min_structures`` and ``mixing_energies``
            (eV per working-ion site, relative to the line between the end points of the sampled
            window). Hull results describe the sampled window only. Raw step
            voltages between consecutive levels in ``voltages`` (V) at ``voltage_x`` (midpoints).
            Convex-hull results from ``InsertionElectrode``: ``voltage_pairs`` (list of dicts with
            ``x_charge``, ``x_discharge``, ``voltage``), ``average_voltage``, ``max_voltage``,
            ``min_voltage`` (V), ``capacity_grav`` (mAh/g), ``capacity_vol`` (mAh/cm^3),
            ``specific_energy`` (Wh/kg), ``energy_density`` (Wh/L) and the ``electrode`` object.
            Also ``ion_reference_energy`` (eV/atom), ``valence``, ``intercalation`` (raw results,
            if ``store_intercalation``) and ``_units``. Merged with relaxation fields from earlier
            steps.

        Raises:
            ValueError: If the host contains no ``species`` sites, ``removal_range`` holds fewer than two
                ion counts, or the ion reference is invalid.
        """
        result = super().calc(structure)
        result, raw_host = self._check_and_prelax(result["final_structure"], result, **_RELAX_DEFAULTS)
        host = to_pmg_structure(raw_host).copy()
        if self.supercell is not None:
            host.make_supercell(self.supercell)
        n_sites = len(host.indices_from_symbol(self.species))
        if n_sites == 0:
            raise ValueError(f"The host structure contains no {self.species} sites")

        n_removed = removal_counts(n_sites, self.removal_range)
        if len(n_removed) < 2:  # noqa: PLR2004
            raise ValueError(
                f"removal_range {tuple(self.removal_range)} contains a single ion count for {n_sites} {self.species} "
                "sites; a voltage needs at least two"
            )
        intercalation_kwargs: dict[str, Any] = {
            # The bounds already fall on whole ions here, so IntercalationCalc does not warn again.
            "removal_range": (n_removed[0] / n_sites, n_removed[-1] / n_sites),
            "species": self.species,
            "algorithm": self.algorithm,
            "relax": self.relax,
            "seed": self.seed,
            "save_freq": max(1, self.nsteps // 10),
            "trajfile": self.trajfile,
            "write_results": False,
            "relax_calc_kwargs": {**_RELAX_DEFAULTS, **self.relax_calc_kwargs} if self.relax else None,
            **self.intercalation_calc_kwargs,
        }
        logger.info("Sampling %d %s levels of a %d-site host", len(n_removed), self.species, len(host))
        levels = IntercalationCalc(self.calculator, self.nsteps, self.temperature, **intercalation_kwargs).calc(host)
        e_ion = self._ion_reference_energy()

        # Sort levels by increasing x (= decreasing number of ions removed).
        ordered = sorted(levels.values(), key=lambda r: -r["Num_removed"])
        removed = np.array([r["Num_removed"] for r in ordered])
        x = (n_sites - removed) / n_sites
        min_energies = np.array([r["min_energy"] for r in ordered], dtype=float)
        # Mixing energy relative to the line between the end points of the sampled window, per site.
        t = (x - x[0]) / (x[-1] - x[0])
        e_line = min_energies[0] + t * (min_energies[-1] - min_energies[0])
        mixing_energies = (min_energies - e_line) / n_sites

        # Raw step voltages: V = [E(fewer ions) + dn * E_ion - E(more ions)] / (dn * valence).
        dn = removed[:-1] - removed[1:]
        voltages = (min_energies[:-1] + dn * e_ion - min_energies[1:]) / (dn * self.valence)

        electrode = InsertionElectrode.from_entries(
            [ComputedStructureEntry(r["min_structure"], r["min_energy"]) for r in ordered],
            ComputedEntry(Composition(self.species), e_ion),
        )
        # pymatgen's x counts ions per framework formula unit. Rescale by the fully intercalated host's
        # value (from its composition, so it holds when the window does not include the full host).
        framework = Composition({el: amt for el, amt in host.composition.items() if el.symbol != self.species})
        x_full = n_sites / framework.get_reduced_composition_and_factor()[1]
        voltage_pairs = [
            {"x_charge": vp.x_charge / x_full, "x_discharge": vp.x_discharge / x_full, "voltage": vp.voltage}
            for vp in electrode.voltage_pairs
        ]

        output: dict[str, Any] = {
            "x": x,
            "x_range": (float(x[0]), float(x[-1])),
            "n_removed": removed,
            "energies": np.array([r["energy"] for r in ordered], dtype=float),
            "min_energies": min_energies,
            "acceptance_ratios": np.array([r["acceptance_ratio"] for r in ordered], dtype=float),
            "min_structures": [r["min_structure"] for r in ordered],
            "mixing_energies": mixing_energies,
            "voltages": voltages,
            "voltage_x": (x[:-1] + x[1:]) / 2,
            "voltage_pairs": voltage_pairs,
            "average_voltage": electrode.get_average_voltage(),
            "max_voltage": electrode.max_voltage,
            "min_voltage": electrode.min_voltage,
            "capacity_grav": electrode.get_capacity_grav(),
            "capacity_vol": electrode.get_capacity_vol(),
            "specific_energy": electrode.get_specific_energy(),
            "energy_density": electrode.get_energy_density(),
            "electrode": electrode,
            "ion_reference_energy": e_ion,
            "valence": self.valence,
        }
        if self.store_intercalation:
            output["intercalation"] = levels
        if self.write_voltage_profile is not None:
            _write_voltage_profile(self.write_voltage_profile, output)

        output["_units"] = self._merge_units(
            result,
            {
                "energies": "eV",
                "min_energies": "eV",
                "mixing_energies": "eV/site",
                "voltages": "V",
                "average_voltage": "V",
                "max_voltage": "V",
                "min_voltage": "V",
                "capacity_grav": "mAh/g",
                "capacity_vol": "mAh/cm^3",
                "specific_energy": "Wh/kg",
                "energy_density": "Wh/L",
                "ion_reference_energy": "eV/atom",
            },
        )
        return result | output

    def _ion_reference_energy(self) -> float:
        """Energy per atom of the working-ion metal reference, computed with ``self.calculator``.

        Returns:
            Reference energy in eV/atom.

        Raises:
            ValueError: If the reference is not a known set, a float, or a pure ``species`` structure.
        """
        if isinstance(self.ion_reference, int | float):
            return float(self.ion_reference)
        if isinstance(self.ion_reference, str):
            path = ELEMENTAL_REFS_DIR / f"{self.ion_reference}-Element-Refs.json.gz"
            if not path.exists():
                raise ValueError(f"Unknown ion_reference {self.ion_reference!r}")
            ref = loadfn(path)[self.species]["structure"]
            candidates = ref if isinstance(ref, list) else [ref]
        else:
            candidates = [self.ion_reference]
        for candidate in candidates:
            if {el.symbol for el in candidate.composition.elements} != {self.species}:
                raise ValueError(f"ion_reference must contain only {self.species}")

        relaxer = self._relaxer(**_RELAX_DEFAULTS) if self.relax else RelaxCalc(self.calculator, max_steps=0)
        energies = []
        for candidate in candidates:
            res = relaxer.calc(candidate)
            energies.append(res["energy"] / len(res["final_structure"]))
        return min(energies)


def _write_voltage_profile(path: str | os.PathLike, output: dict[str, Any]) -> None:
    """Write per-level energies and the step voltage to the next level (higher x) as CSV.

    Args:
        path: Output file path.
        output: The ``VoltageCalc`` result dict.
    """
    fields = ["x", "n_removed", "energy_eV", "min_energy_eV", "acceptance_ratio", "mixing_energy_eV_per_site"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*fields, "step_voltage_V"])
        for i in range(len(output["x"])):
            step = output["voltages"][i] if i < len(output["voltages"]) else ""
            writer.writerow(
                [
                    output["x"][i],
                    output["n_removed"][i],
                    output["energies"][i],
                    output["min_energies"][i],
                    output["acceptance_ratios"][i],
                    output["mixing_energies"][i],
                    step,
                ]
            )
