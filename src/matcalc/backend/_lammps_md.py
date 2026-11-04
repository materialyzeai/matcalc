"""Molecular dynamics with the LAMMPS backend.

``run_lammps_md`` runs the ensembles of ``MDCalc`` with the equivalent LAMMPS fixes in the same in-process LAMMPS
instance as ``run_lammps``. Initial velocities come from the input atoms, so both backends start from identical
conditions. The NVE integrator is the same velocity Verlet scheme as ASE's; thermostats and barostats sample the same
ensembles with different integrators (e.g. LAMMPS ``fix npt`` is MTK, ASE ``NPT`` is Melchionna), so trajectories only
agree statistically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from ._lammps import (
    ASE_TIME_TO_PS,
    ASE_VELOCITY_TO_LAMMPS,
    BAR_TO_EVA3,
    get_engine,
    get_lammps_potential,
)

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from ._lammps import _Frame, _LAMMPSEngine

_FIX_IDS = ("matcalc_md", "matcalc_thermostat", "matcalc_barostat")
_ORTHO_ENSEMBLES = ("npt_berendsen", "npt_inhomogeneous")


def run_lammps_md(
    atoms: Atoms,
    calculator: Calculator,
    *,
    ensemble: str,
    temperature: float,
    timestep: float,
    steps: int,
    loginterval: int = 1,
    pressure: float = 0.0,
    taut: float,
    taup: float,
    friction: float = 1.0e-3,
    ttime: float = 25.0,
    fixed_cell: bool = False,
    external_stress: float | np.ndarray | None = None,
    compressibility_au: float | None = None,
    mask: Any = None,
    tchain: int = 3,
    pchain: int = 3,
    tloop: int = 1,
    ploop: int = 1,
    seed: int | None = None,
    logfile: str | None = None,
) -> list[Atoms]:
    """Run MD in LAMMPS with the force field of ``calculator``; parameters follow ``MDCalc`` and ASE units.

    Args:
        atoms: Initial structure with velocities.
        calculator: ASE calculator with a LAMMPS representation (see ``run_lammps``).
        ensemble: ``MDCalc`` ensemble name. ``nvt_andersen`` has no LAMMPS equivalent.
        temperature: Target temperature (K).
        timestep: Time step (fs).
        steps: Number of MD steps.
        loginterval: Interval between returned frames.
        pressure: Target pressure (eV/Å^3) for the Berendsen and MTK barostats.
        taut: Thermostat time constant (ASE time units).
        taup: Barostat time constant (ASE time units); also used for ``npt``/``npt_nose_hoover``, whose ASE
            ``pfactor`` has no LAMMPS equivalent.
        friction: Langevin friction (1 / ASE time units).
        ttime: Thermostat time for ``npt``/``npt_nose_hoover`` (fs).
        fixed_cell: For ``npt``/``npt_nose_hoover``, keep the cell fixed (Nose-Hoover NVT with ``ttime``), as ASE
            ``NPT`` does with ``pfactor=None``.
        external_stress: Target pressure (eV/Å^3, scalar) for ``npt``/``npt_nose_hoover``.
        compressibility_au: Compressibility (Å^3/eV) for the Berendsen barostats; required, as in ASE.
        mask: Cell mask for ``npt``/``npt_nose_hoover``: None or diagonal (orthogonal cell changes, LAMMPS ``aniso``)
            or all ones (fully flexible cell, LAMMPS ``tri``).
        tchain: Thermostat chain length (MTK ensembles).
        pchain: Barostat chain length (MTK ensembles).
        tloop: Thermostat sub-steps (MTK ensembles).
        ploop: Barostat sub-steps (MTK ensembles).
        seed: Seed for stochastic thermostats.
        logfile: Optional LAMMPS log file with thermo output every ``loginterval`` steps.

    Returns:
        Frames at steps 0, loginterval, 2 * loginterval, ..., as ASE MD observers record them, each with velocities
        and a SinglePointCalculator holding energy, forces and stress. Positions are unwrapped.
    """
    engine = get_engine(get_lammps_potential(calculator), atoms)
    lmp = engine.lmp
    masses = _type_masses(atoms, engine)
    frame = engine.load(atoms, velocities=atoms.get_velocities() * ASE_VELOCITY_TO_LAMMPS)
    fixes = _ensemble_fixes(
        ensemble,
        temperature=temperature,
        taut=taut * ASE_TIME_TO_PS,
        taup=taup * ASE_TIME_TO_PS,
        damp=ASE_TIME_TO_PS / friction,
        ttime=ttime * 1e-3,
        fixed_cell=fixed_cell,
        pressure=pressure / BAR_TO_EVA3,
        external_stress=external_stress,
        compressibility_au=compressibility_au,
        mask=mask,
        chains=f"tchain {tchain} pchain {pchain} tloop {tloop} ploop {ploop}",
        seed=seed,
    )
    setup = [*masses, "reset_timestep 0", f"timestep {timestep * 1e-3}"]
    cleanup = []
    if ensemble.lower() in _ORTHO_ENSEMBLES:
        # fix press/berendsen does not support triclinic boxes.
        cell = engine.cell()
        if np.abs(cell[np.tril_indices(3, -1)]).max() > 1e-8 * np.abs(np.diag(cell)).max():
            raise NotImplementedError(f"{ensemble} with the LAMMPS backend requires an orthogonal cell.")
        setup.append("change_box all xy final 0.0 xz final 0.0 yz final 0.0 ortho")
        cleanup.append("change_box all triclinic")
    setup += fixes
    if logfile:
        setup += [f'log "{logfile}"', f"thermo {loginterval}"]
        cleanup += ["log none", "thermo 0"]

    lmp.commands_list(setup)
    try:
        lmp.command("run 0")
        image, frac = _snapshot(engine, frame, atoms, frame.ref_frac)
        frames = [image]
        for _ in range(steps // loginterval):
            lmp.command(f"run {loginterval} pre no post no")
            image, frac = _snapshot(engine, frame, atoms, frac)
            frames.append(image)
        if steps % loginterval:
            lmp.command(f"run {steps % loginterval} pre no post no")
    finally:
        lmp.commands_list(
            [f"unfix {fix}" for fix in _FIX_IDS if any(cmd.startswith(f"fix {fix} ") for cmd in fixes)] + cleanup
        )
    return frames


def _type_masses(atoms: Atoms, engine: _LAMMPSEngine) -> list[str]:
    """``mass`` commands so that LAMMPS uses the masses of ``atoms`` (one mass per element)."""
    commands = []
    masses = atoms.get_masses()
    symbols = np.array(atoms.get_chemical_symbols())
    for el in sorted(set(symbols)):
        el_masses = masses[symbols == el]
        if not np.allclose(el_masses, el_masses[0]):
            raise NotImplementedError(f"The LAMMPS backend needs one mass per element; {el} atoms have several.")
        commands.append(f"mass {engine.type_of[el]} {el_masses[0]}")
    return commands


def _ensemble_fixes(  # noqa: C901, PLR0911, PLR0912
    ensemble: str,
    *,
    temperature: float,
    taut: float,
    taup: float,
    damp: float,
    ttime: float,
    fixed_cell: bool,
    pressure: float,
    external_stress: float | np.ndarray | None,
    compressibility_au: float | None,
    mask: Any,
    chains: str,
    seed: int | None,
) -> list[str]:
    """LAMMPS fixes for an ``MDCalc`` ensemble; times in ps, pressures in bar."""
    t = f"{temperature} {temperature}"
    nve = "fix matcalc_md all nve"
    rng = np.random.default_rng(seed)
    lmp_seed = int(rng.integers(1, 900_000_000))  # LAMMPS' Marsaglia RNG limit
    ensemble = ensemble.lower()

    if ensemble == "nve":
        return [nve]
    if ensemble in ("nvt", "nvt_nose_hoover"):
        return [f"fix matcalc_md all nvt temp {t} {taut} tchain 3"]
    if ensemble == "nvt_berendsen":
        return [nve, f"fix matcalc_thermostat all temp/berendsen {t} {taut}"]
    if ensemble == "nvt_langevin":
        return [nve, f"fix matcalc_thermostat all langevin {t} {damp} {lmp_seed} zero yes"]
    if ensemble == "nvt_bussi":
        return [nve, f"fix matcalc_thermostat all temp/csvr {t} {taut} {lmp_seed}"]
    if ensemble == "nvt_andersen":
        raise NotImplementedError("LAMMPS has no Andersen thermostat; use MATCALC_BACKEND=ASE for nvt_andersen.")
    if ensemble in ("npt", "npt_nose_hoover"):
        if fixed_cell:
            return [f"fix matcalc_md all nvt temp {t} {ttime}"]
        if external_stress is None:
            target = 0.0
        elif np.ndim(external_stress) == 0:
            target = float(external_stress) / BAR_TO_EVA3  # type: ignore[arg-type]
        else:
            raise NotImplementedError("The LAMMPS backend supports a scalar external_stress (pressure) only.")
        return [f"fix matcalc_md all npt temp {t} {ttime} {_npt_coupling(mask)} {target} {target} {taup}"]
    if ensemble in ("npt_berendsen", "npt_inhomogeneous"):
        if compressibility_au is None:
            raise ValueError(f"{ensemble} requires compressibility_au.")
        modulus = 1 / compressibility_au / BAR_TO_EVA3
        coupling = "iso" if ensemble == "npt_berendsen" else "aniso"
        return [
            nve,
            f"fix matcalc_thermostat all temp/berendsen {t} {taut}",
            f"fix matcalc_barostat all press/berendsen {coupling} {pressure} {pressure} {taup} modulus {modulus}",
        ]
    if ensemble in ("npt_mtk", "npt_isotropic_mtk"):
        coupling = "tri" if ensemble == "npt_mtk" else "iso"
        return [f"fix matcalc_md all npt temp {t} {taut} {coupling} {pressure} {pressure} {taup} {chains}"]
    raise ValueError(f"Unsupported ensemble {ensemble!r}.")


def _npt_coupling(mask: Any) -> str:
    """LAMMPS barostat coupling for an ASE ``NPT`` cell mask."""
    if mask is None:
        return "aniso"
    mask = np.asarray(mask)
    if mask.shape == (3,):
        mask = np.diag(mask)
    if np.array_equal(mask, np.eye(3)):
        return "aniso"
    if np.all(mask == 1):
        return "tri"
    raise NotImplementedError("The LAMMPS backend supports NPT masks that are diagonal ones or all ones only.")


def _snapshot(engine: _LAMMPSEngine, frame: _Frame, atoms: Atoms, ref_frac: np.ndarray) -> tuple[Atoms, np.ndarray]:
    """Current LAMMPS state as ASE Atoms in the input frame, and its fractional coordinates.

    Positions are unwrapped to the periodic images closest to ``ref_frac`` (the previous frame).
    """
    cell = engine.cell()
    positions, frac = frame.positions(engine.gather("x"), cell, ref_frac)
    image = Atoms(
        atoms.get_chemical_symbols(),
        positions=positions,
        cell=frame.cell(cell),
        pbc=True,
        masses=atoms.get_masses(),
    )
    image.set_velocities(frame.vectors(engine.gather("v")) / ASE_VELOCITY_TO_LAMMPS)
    energy = engine.energy()
    image.calc = SinglePointCalculator(
        image,
        energy=energy,
        free_energy=energy,
        forces=frame.vectors(engine.gather("f")),
        stress=full_3x3_to_voigt_6_stress(frame.tensor(engine.stress())),
    )
    return image, frac
