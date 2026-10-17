"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import difflib
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .units import eVA3ToGPa

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from maml.apps.pes import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
    from pymatgen.core import IMolecule, IStructure


# Unified naming convention for foundation potentials:
#
#     <Architecture>-<Dataset>[-<Functional>]-<Version>[-<Variant>]
#
# e.g. ``TensorNet-MatPES-PBE-2025.2`` or ``MACE-MPA-0-medium``. Each entry in
# ``MODEL_REGISTRY`` maps a canonical name to a provider and the provider-specific
# kwargs used to materialise the calculator. Users select a model by its canonical
# name; ``MODEL_ALIASES`` provides short / legacy spellings that resolve to a
# canonical name.
#
# Dataset labels:
# Label                      | Data                  | Notes
# MPtrj                      | MPtrj                 | Materials Project relaxation trajectories.
# MPA                        | MPtrj + Alexandria    | MACE-MPA-0 uses sAlex; ORB-v2 uses Alexandria.
# OMat24                     | OMat24                | Open Materials 2024.
# OAM                        | OMat24 + sAlex + MPtrj | GRACE and SevenNet use different training procedures.
# MatPES-PBE / MatPES-r2SCAN  | MatPES                | The suffix identifies the functional.
# SMAX                       | SMAX                  | GRACE max-entropy structures; element-dependent 5.0-7.5 A cutoff.
# SMAX-OMat24                | SMAX + OMat24         | GRACE's general-purpose recommendation.
# MAD                        | MAD                   | PET-MAD; MAD-1.0 is PBEsol, MAD-1.5/1.6 are r2SCAN.
# Omni                       | SevenNet-Omni data    | Multi-domain; the registry uses modal="mpa".
# OpenLAM                    | OpenLAM               | Multi-task DPA3 pretraining; the registry uses the Omat24 head.
#
# GRACE OAM pretrains on OMat24, then fine-tunes on sAlex + MPtrj. SevenNet OAM
# uses joint multifidelity training; modal="mpa" selects the MPtrj/sAlex DFT setting.
# sAlex is an Alexandria subset with WBM matches removed and trajectories subsampled.
#
# Exceptions to the convention, where upstream names lack a field:
#   - UMA uses ``<Architecture>-<Size>-<Version>`` and the ``omat`` task (no dataset field).
#   - MACE-MH-0/1 keep the upstream names (multi-head models; the registry uses the ``omat_pbe`` head).
#   - SevenNet OAM/OMat24/Omni, GRACE and QET-MatQ names omit a version.
#   - MatterSim keeps its upstream names (``MatterSim-v1.0.0-1M``/``-5M``), omitting the M3GNet architecture.
#
# To add a new model, append an entry here. The ``provider`` key picks the loader
# branch in ``PESCalculator.load_universal``; remaining keys are forwarded to that
# loader (user kwargs to ``load_universal`` override these defaults).
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # MatGL — TensorNet on MatPES
    "TensorNet-MatPES-PBE-2025.2": {"provider": "matgl", "path": "TensorNet-PES-MatPES-PBE-2025.2"},
    "TensorNet-MatPES-r2SCAN-2025.2": {"provider": "matgl", "path": "TensorNet-PES-MatPES-r2SCAN-2025.2"},
    # MatGL — M3GNet
    "M3GNet-MatPES-PBE-2025.2": {"provider": "matgl", "path": "M3GNet-PES-MatPES-PBE-2025.2"},
    "M3GNet-MatPES-r2SCAN-2025.2": {"provider": "matgl", "path": "M3GNet-PES-MatPES-r2SCAN-2025.2"},
    # MatGL — QET
    "QET-MatPES-PBE-2025.2": {"provider": "matgl", "path": "QET-PES-MatPES-PBE-2025.2"},
    "QET-MatPES-r2SCAN-2025.2": {"provider": "matgl", "path": "QET-PES-MatPES-r2SCAN-2025.2"},
    "QET-MatQ": {"provider": "matgl", "path": "QET-PES-MatQ"},
    # MatGL — CHGNet
    "CHGNet-MatPES-PBE-2026.9-1M": {"provider": "matgl", "path": "CHGNet-PES-MatPES-PBE-1M-2026.9"},
    "CHGNet-MatPES-r2SCAN-2026.9-1M": {"provider": "matgl", "path": "CHGNet-PES-MatPES-r2SCAN-1M-2026.9"},
    "CHGNet-MatPES-PBE-2025.2.10": {"provider": "matgl", "path": "CHGNet-PES-MatPES-PBE-2025.2.10"},
    "CHGNet-MatPES-r2SCAN-2025.2.10": {"provider": "matgl", "path": "CHGNet-PES-MatPES-r2SCAN-2025.2.10"},
    # MACE foundation models (mace-foundations release names)
    "MACE-MPtrj-0a-small": {"provider": "mace_mp", "model": "small"},
    "MACE-MPtrj-0a-medium": {"provider": "mace_mp", "model": "medium"},
    "MACE-MPtrj-0a-large": {"provider": "mace_mp", "model": "large"},
    "MACE-MPtrj-0b-small": {"provider": "mace_mp", "model": "small-0b"},
    "MACE-MPtrj-0b-medium": {"provider": "mace_mp", "model": "medium-0b"},
    "MACE-MPtrj-0b2-small": {"provider": "mace_mp", "model": "small-0b2"},
    "MACE-MPtrj-0b2-medium": {"provider": "mace_mp", "model": "medium-0b2"},
    "MACE-MPtrj-0b2-large": {"provider": "mace_mp", "model": "large-0b2"},
    "MACE-MPtrj-0b3-medium": {"provider": "mace_mp", "model": "medium-0b3"},
    "MACE-MPA-0-medium": {"provider": "mace_mp", "model": "medium-mpa-0"},
    "MACE-OMat24-0-small": {"provider": "mace_mp", "model": "small-omat-0"},
    "MACE-OMat24-0-medium": {"provider": "mace_mp", "model": "medium-omat-0"},
    "MACE-MatPES-PBE-0": {"provider": "mace_mp", "model": "mace-matpes-pbe-0"},
    "MACE-MatPES-r2SCAN-0": {"provider": "mace_mp", "model": "mace-matpes-r2scan-0"},
    "MACE-MH-0": {"provider": "mace_mp", "model": "mh-0", "head": "omat_pbe"},
    "MACE-MH-1": {"provider": "mace_mp", "model": "mh-1", "head": "omat_pbe"},
    # NOTE: MACE multi-head checkpoints (mh-0, mh-1) require a mandatory ``head=...``
    # kwarg per use; they are not registered as default-loadable canonical names.
    # SevenNet (provider arguments use upstream SevenNetCalculator names)
    "SevenNet-MPtrj-0": {"provider": "sevennet", "model": "7net-0"},
    "SevenNet-MPtrj-l3i5": {"provider": "sevennet", "model": "7net-l3i5"},
    "SevenNet-OAM-MF": {"provider": "sevennet", "model": "7net-mf-ompa", "modal": "mpa"},
    "SevenNet-OMat24": {"provider": "sevennet", "model": "7net-omat"},
    "SevenNet-Omni": {"provider": "sevennet", "model": "7net-omni", "modal": "mpa"},
    "SevenNet-Omni-i8": {"provider": "sevennet", "model": "7net-omni-i8", "modal": "mpa"},
    "SevenNet-Omni-i12": {"provider": "sevennet", "model": "7net-omni-i12", "modal": "mpa"},
    # GRACE / TensorPotential
    "GRACE-SMAX-OMat24-1L-large": {"provider": "grace", "model": "GRACE-1L-SMAX-OMAT-large"},
    "GRACE-SMAX-OMat24-2L-medium": {"provider": "grace", "model": "GRACE-2L-SMAX-OMAT-medium"},
    "GRACE-SMAX-OMat24-2L-large": {"provider": "grace", "model": "GRACE-2L-SMAX-OMAT-large"},
    "GRACE-SMAX-1L-large": {"provider": "grace", "model": "GRACE-1L-SMAX-large"},
    "GRACE-SMAX-2L-medium": {"provider": "grace", "model": "GRACE-2L-SMAX-medium"},
    "GRACE-SMAX-2L-large": {"provider": "grace", "model": "GRACE-2L-SMAX-large"},
    "GRACE-OAM-1L-small": {"provider": "grace", "model": "GRACE-1L-OAM"},
    "GRACE-OAM-1L-medium": {"provider": "grace", "model": "GRACE-1L-OMAT-medium-ft-AM"},
    "GRACE-OAM-1L-large": {"provider": "grace", "model": "GRACE-1L-OMAT-large-ft-AM"},
    "GRACE-OAM-2L-small": {"provider": "grace", "model": "GRACE-2L-OAM"},
    "GRACE-OAM-2L-medium": {"provider": "grace", "model": "GRACE-2L-OMAT-medium-ft-AM"},
    "GRACE-OAM-2L-large": {"provider": "grace", "model": "GRACE-2L-OMAT-large-ft-AM"},
    "GRACE-OAM-3L-large": {"provider": "grace", "model": "GRACE-3L-OMAT-large-ft-AM"},
    "GRACE-OMat24-1L-medium-base": {"provider": "grace", "model": "GRACE-1L-OMAT-medium-base"},
    "GRACE-OMat24-1L-medium-ft-E": {"provider": "grace", "model": "GRACE-1L-OMAT-medium-ft-E"},
    "GRACE-OMat24-1L-large-base": {"provider": "grace", "model": "GRACE-1L-OMAT-large-base"},
    "GRACE-OMat24-1L-large-ft-E": {"provider": "grace", "model": "GRACE-1L-OMAT-large-ft-E"},
    "GRACE-OMat24-2L-small": {"provider": "grace", "model": "GRACE-2L-OMAT"},
    "GRACE-OMat24-2L-medium-base": {"provider": "grace", "model": "GRACE-2L-OMAT-medium-base"},
    "GRACE-OMat24-2L-medium-ft-E": {"provider": "grace", "model": "GRACE-2L-OMAT-medium-ft-E"},
    "GRACE-OMat24-2L-large-base": {"provider": "grace", "model": "GRACE-2L-OMAT-large-base"},
    "GRACE-OMat24-2L-large-ft-E": {"provider": "grace", "model": "GRACE-2L-OMAT-large-ft-E"},
    "GRACE-OMat24-3L-large": {"provider": "grace", "model": "GRACE-3L-OMAT-large"},
    # GRACE MP models are only available in tensorpotential<0.5.8; -r5/-r6 give the cutoff in Angstrom.
    "GRACE-MPtrj-1L-r6": {"provider": "grace", "model": "GRACE-1L-MP-r6"},
    "GRACE-MPtrj-2L-r5": {"provider": "grace", "model": "GRACE-2L-MP-r5"},
    "GRACE-MPtrj-2L-r6": {"provider": "grace", "model": "GRACE-2L-MP-r6"},
    # Orb (orb-models>=0.6 needs Python>=3.12)
    "ORB-MPA-v2": {"provider": "orb", "model": "orb-v2"},
    "ORB-MPA-v2-D3": {"provider": "orb", "model": "orb-d3-v2"},
    "ORB-MPA-v2-D3-sm": {"provider": "orb", "model": "orb-d3-sm-v2"},
    "ORB-MPA-v2-D3-xs": {"provider": "orb", "model": "orb-d3-xs-v2"},
    "ORB-MPtrj-v2": {"provider": "orb", "model": "orb-mptraj-only-v2"},
    "ORB-MPA-v3-conservative-20": {"provider": "orb", "model": "orb-v3-conservative-20-mpa"},
    "ORB-MPA-v3-conservative-inf": {"provider": "orb", "model": "orb-v3-conservative-inf-mpa"},
    "ORB-MPA-v3-direct-20": {"provider": "orb", "model": "orb-v3-direct-20-mpa"},
    "ORB-MPA-v3-direct-inf": {"provider": "orb", "model": "orb-v3-direct-inf-mpa"},
    "ORB-OMat24-v3-conservative-20": {"provider": "orb", "model": "orb-v3-conservative-20-omat"},
    "ORB-OMat24-v3-conservative-inf": {"provider": "orb", "model": "orb-v3-conservative-inf-omat"},
    "ORB-OMat24-v3-direct-20": {"provider": "orb", "model": "orb-v3-direct-20-omat"},
    "ORB-OMat24-v3-direct-inf": {"provider": "orb", "model": "orb-v3-direct-inf-omat"},
    # MatterSim (M3GNet architecture)
    "MatterSim-v1.0.0-1M": {"provider": "mattersim", "load_path": "MatterSim-v1.0.0-1M.pth"},
    "MatterSim-v1.0.0-5M": {"provider": "mattersim", "load_path": "MatterSim-v1.0.0-5M.pth"},
    # FAIRChem (UMA family) — upstream uses ``uma-s-1p2`` style; we expose them
    # under ``<Architecture>-<Size>-<Version>`` with the ``omat`` task.
    "UMA-S-1.2.1": {"provider": "fairchem", "model": "uma-s-1p2p1", "task_name": "omat"},
    "UMA-S-1.2": {"provider": "fairchem", "model": "uma-s-1p2", "task_name": "omat"},
    "UMA-S-1.1": {"provider": "fairchem", "model": "uma-s-1p1", "task_name": "omat"},
    "UMA-M-1.1": {"provider": "fairchem", "model": "uma-m-1p1", "task_name": "omat"},
    # UPET
    "PET-OMat24-1.0.0-XS": {"provider": "upet", "model": "pet-omat-xs", "version": "1.0.0"},
    "PET-OMat24-1.0.0-S": {"provider": "upet", "model": "pet-omat-s", "version": "1.0.0"},
    "PET-OMat24-1.0.0-M": {"provider": "upet", "model": "pet-omat-m", "version": "1.0.0"},
    "PET-OMat24-1.0.0-L": {"provider": "upet", "model": "pet-omat-l", "version": "1.0.0"},
    "PET-OMat24-1.0.0-XL": {"provider": "upet", "model": "pet-omat-xl", "version": "1.0.0"},
    "PET-OAM-0.1.0-L": {"provider": "upet", "model": "pet-oam-l", "version": "0.1.0"},
    "PET-OAM-1.0.0-XL": {"provider": "upet", "model": "pet-oam-xl", "version": "1.0.0"},
    "PET-MatPES-r2SCAN-0.1.0-L": {"provider": "upet", "model": "pet-omatpes-l", "version": "0.1.0"},
    "PET-MAD-r2SCAN-1.6.0-XS": {"provider": "upet", "model": "pet-mad-xs", "version": "1.6.0"},
    "PET-MAD-r2SCAN-1.6.0-S": {"provider": "upet", "model": "pet-mad-s", "version": "1.6.0"},
    "PET-MAD-r2SCAN-1.6.0-M": {"provider": "upet", "model": "pet-mad-m", "version": "1.6.0"},
    "PET-MAD-PBEsol-1.0.2-S": {"provider": "upet", "model": "pet-mad-s", "version": "1.0.2"},
    # DPA (DPA4 checkpoints are CC-BY-NC-4.0)
    # DPA-2.4-7M targets deepmd-kit 3.1 (upstream model card).
    "DPA2-OpenLAM-2.4-7M": {"provider": "deepmd", "model": "DPA-2.4-7M", "head": "Omat24"},
    "DPA3-OpenLAM-3.1-3M": {"provider": "deepmd", "model": "DPA-3.1-3M", "head": "Omat24"},
    "DPA3-OpenLAM-3.2-5M": {"provider": "deepmd", "model": "DPA-3.2-5M", "head": "Omat24"},
    "DPA3-OpenLAM-3.3-1M": {"provider": "deepmd", "model": "DPA-3.3-1M", "head": "Omat24"},
    "DPA4-OMat24-2026.8.5-Nano": {"provider": "deepmd", "model": "DPA4-Nano-OMat24-v20260805"},
    "DPA4-OMat24-2026.8.5-Mini": {"provider": "deepmd", "model": "DPA4-Mini-OMat24-v20260805"},
    "DPA4-OMat24-2026.8.5-Neo": {"provider": "deepmd", "model": "DPA4-Neo-OMat24-v20260805"},
    "DPA4-OMat24-2026.8.5-Air": {"provider": "deepmd", "model": "DPA4-Air-OMat24-v20260805"},
    "DPA4-OMat24-2026.8.5-Plus": {"provider": "deepmd", "model": "DPA4-Plus-OMat24-v20260805"},
}

# Short / legacy aliases. Keys are matched case-insensitively. Values must be
# canonical names from ``MODEL_REGISTRY``.
MODEL_ALIASES: dict[str, str] = {
    # short architecture / functional aliases — pick a sensible default per family
    "tensornet": "TensorNet-MatPES-PBE-2025.2",
    "m3gnet": "M3GNet-MatPES-PBE-2025.2",
    "chgnet": "CHGNet-MatPES-PBE-2026.9-1M",
    "qet": "QET-MatPES-PBE-2025.2",
    "pbe": "TensorNet-MatPES-PBE-2025.2",
    "r2scan": "TensorNet-MatPES-r2SCAN-2025.2",
    "mace": "MACE-MPA-0-medium",
    "sevennet": "SevenNet-Omni",
    "grace": "GRACE-SMAX-OMat24-2L-medium",
    "tensorpotential": "GRACE-SMAX-OMat24-2L-medium",
    "orb": "ORB-OMat24-v3-conservative-inf",
    "mattersim": "MatterSim-v1.0.0-1M",
    "fairchem": "UMA-S-1.2.1",
    "uma": "UMA-S-1.2.1",
    "pet": "PET-MAD-r2SCAN-1.6.0-S",
    "petmad": "PET-MAD-PBEsol-1.0.2-S",
    "deepmd": "DPA4-OMat24-2026.8.5-Mini",
    "dpa": "DPA4-OMat24-2026.8.5-Mini",
    # Previous canonical names.
    "mace-mp-0-small": "MACE-MPtrj-0a-small",
    "mace-mp-0-medium": "MACE-MPtrj-0a-medium",
    "mace-mp-0-large": "MACE-MPtrj-0a-large",
    "mace-mp-0b-small": "MACE-MPtrj-0b-small",
    "mace-mp-0b-medium": "MACE-MPtrj-0b-medium",
    "mace-mp-0b2-small": "MACE-MPtrj-0b2-small",
    "mace-mp-0b2-medium": "MACE-MPtrj-0b2-medium",
    "mace-mp-0b2-large": "MACE-MPtrj-0b2-large",
    "mace-mp-0b3-medium": "MACE-MPtrj-0b3-medium",
    "mace-omat-0-small": "MACE-OMat24-0-small",
    "mace-omat-0-medium": "MACE-OMat24-0-medium",
    "grace-1l-oam": "GRACE-OAM-1L-small",
    "grace-2l-oam": "GRACE-OAM-2L-small",
    "grace-2l-omat": "GRACE-OMat24-2L-small",
    "grace-2l-mptrj": "GRACE-MPtrj-2L-r6",
    "orb-v2": "ORB-MPA-v2",
    "orb-d3-v2": "ORB-MPA-v2-D3",
    "orb-d3-sm-v2": "ORB-MPA-v2-D3-sm",
    "orb-d3-xs-v2": "ORB-MPA-v2-D3-xs",
    "sevennet-0": "SevenNet-MPtrj-0",
    "sevennet-l3i5": "SevenNet-MPtrj-l3i5",
    "sevennet-mf-ompa": "SevenNet-OAM-MF",
    "sevennet-omat": "SevenNet-OMat24",
    # legacy MatGL spellings — keep working so existing code/notebooks don't break
    "tensornet-pes-matpes-pbe-2025.2": "TensorNet-MatPES-PBE-2025.2",
    "tensornet-pes-matpes-r2scan-2025.2": "TensorNet-MatPES-r2SCAN-2025.2",
    "m3gnet-pes-matpes-pbe-2025.2": "M3GNet-MatPES-PBE-2025.2",
    "m3gnet-pes-matpes-r2scan-2025.2": "M3GNet-MatPES-r2SCAN-2025.2",
    "qet-pes-matpes-pbe-2025.2": "QET-MatPES-PBE-2025.2",
    "qet-pes-matpes-r2scan-2025.2": "QET-MatPES-r2SCAN-2025.2",
    "qet-pes-matq": "QET-MatQ",
    "chgnet-pes-matpes-pbe-1m-2026.9": "CHGNet-MatPES-PBE-2026.9-1M",
    "chgnet-pes-matpes-r2scan-1m-2026.9": "CHGNet-MatPES-r2SCAN-2026.9-1M",
    "chgnet-matpes-pbe-1m-2026.9": "CHGNet-MatPES-PBE-2026.9-1M",
    "chgnet-matpes-r2scan-1m-2026.9": "CHGNet-MatPES-r2SCAN-2026.9-1M",
    "chgnet-pes-matpes-pbe-2025.2.10": "CHGNet-MatPES-PBE-2025.2.10",
    "chgnet-pes-matpes-r2scan-2025.2.10": "CHGNet-MatPES-r2SCAN-2025.2.10",
    "chgnet-matpes-pbe-2025.2.10-2.7m-pes": "CHGNet-MatPES-PBE-2025.2.10",
}

try:
    # Set of raw MatGL pretrained PES names. Used as a backward-compat escape hatch
    # in ``load_universal`` for matgl models not yet registered as canonical names.
    import matgl

    _MATGL_AVAILABLE: set[str] = {
        m for m in matgl.get_available_pretrained_models() if "PES" in m and "ANI-1x-Subset-PES" not in m
    }
except ImportError:
    _MATGL_AVAILABLE = set()
except Exception as _matgl_err:  # noqa: BLE001
    warnings.warn(
        f"Unable to query pre-trained MatGL universal calculators: {_matgl_err!r}",
        stacklevel=1,
    )
    _MATGL_AVAILABLE = set()

UNIVERSAL_CALCULATORS = Enum(  # type: ignore[misc]
    "UNIVERSAL_CALCULATORS", {k: k for k in sorted(MODEL_REGISTRY)}
)

# Same strings as enum values; exposed for typing-friendly iteration (e.g. CLI choices).
UNIVERSAL_CALCULATOR_NAMES: tuple[str, ...] = tuple(sorted(MODEL_REGISTRY))

# Case-folded view of MODEL_REGISTRY so users typing the wrong case still resolve
# to the right canonical name (e.g. ``tensornet-matpes-pbe-2025.2`` → canonical).
_REGISTRY_LOWER: dict[str, str] = {k.lower(): k for k in MODEL_REGISTRY}

# Pip-install hint surfaced when a provider's dependency is not installed.
_PROVIDER_INSTALL_EXTRA: dict[str, str] = {
    "matgl": "matgl",
    "mace_mp": "mace",
    "sevennet": "sevennet",
    "grace": "grace",
    "orb": "orb",
    "mattersim": "mattersim",
    "fairchem": "fairchem",
    "upet": "upet",
    "deepmd": "deepmd",
}


def _install_hint(provider: str) -> str:
    """Return a human-friendly install hint for a missing provider dependency."""
    extra = _PROVIDER_INSTALL_EXTRA.get(provider, provider)
    return (
        f"Provider {provider!r} requires an optional dependency that is not installed. "
        f"Install it with: pip install 'matcalc[{extra}]'"
    )


def _resolve_canonical(name: str) -> str | None:
    """Resolve ``name`` (alias, canonical, or case-variant) to a canonical model name.

    Returns ``None`` if the name cannot be resolved.
    """
    # 1. Direct hit on canonical registry (preserves case).
    if name in MODEL_REGISTRY:
        return name
    # 2. Case-insensitive alias resolution.
    alias_target = MODEL_ALIASES.get(name.lower())
    if alias_target is not None:
        return alias_target
    # 3. Case-insensitive canonical lookup (recovers from common typos like
    #    ``tensornet-matpes-pbe-2025.2``).
    case_folded = _REGISTRY_LOWER.get(name.lower())
    if case_folded is not None:
        return case_folded
    return None


def _suggest_models(name: str, n: int = 3) -> list[str]:
    """Return the closest candidate model names for an unrecognised input."""
    candidates = list(MODEL_REGISTRY) + list(MODEL_ALIASES)
    return difflib.get_close_matches(name, candidates, n=n, cutoff=0.5)


class PESCalculator(Calculator):
    """
    Class for simulating and calculating potential energy surfaces (PES) using various
    machine learning and classical potentials. It extends the ASE `Calculator` API,
    allowing integration with the ASE framework for molecular dynamics and structure
    optimization.

    PESCalculator provides methods to perform energy, force, and stress calculations
    using potentials such as MTP, GAP, NNP, SNAP, ACE, NequIP, DeePMD and MatGL (M3GNet, TensorNet, CHGNet). The class
    includes utilities to load compatible models for each potential type, making it
    a versatile tool for materials modeling and molecular simulations.

    Attributes:
        potential: MAML LAMMPS static potential backend.
        stress_weight: Factor applied to stress (includes unit conversion).
    """

    implemented_properties = ["energy", "forces", "stress"]  # noqa:RUF012

    def __init__(
        self,
        potential: LMPStaticCalculator,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PESCalculator with a potential from maml.

        Args:
            potential: MAML ``LMPStaticCalculator`` instance.
            stress_unit: ``"GPa"`` or ``"eV/A3"`` for returned stress units.
            stress_weight: Multiplier on stress after unit conversion (default 1.0).
            **kwargs: Forwarded to ``ase.calculators.calculator.Calculator``.
        """
        super().__init__(**kwargs)
        self.potential = potential

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = 1 / eVA3ToGPa  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.stress_weight = stress_weight * conversion_factor

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """
        Perform calculation for an input Atoms.

        Args:
            atoms: Structure to evaluate.
            properties: ASE property list to compute (defaults to all).
            system_changes: ASE change list; if unchanged, cached results may be reused.
        """
        from ase.calculators.calculator import all_changes, all_properties
        from maml.apps.pes import EnergyForceStress
        from pymatgen.io.ase import AseAtomsAdaptor

        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        structure: Structure | IStructure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore[arg-type,assignment]
        efs_calculator = EnergyForceStress(ff_settings=self.potential)
        energy, forces, stresses = efs_calculator.calculate([structure])[0]

        self.results = {
            "energy": energy,
            "forces": forces,
            "stress": stresses * self.stress_weight,
        }

    @staticmethod
    def load_matgl(path: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a MATGL model from the specified path and initializes a PESCalculator
        with the loaded model and additional optional parameters.

        This method uses the MATGL library to load a model from the given file path
        or directory. It then configures a calculator using the loaded model and
        the provided keyword arguments.

        Args:
            path: Path to the MatGL model file or pretrained model name.
            **kwargs: Forwarded to the MatGL ASE calculator.

        Returns:
            Configured ASE calculator for the MatGL model.
        """
        import matgl

        model = matgl.load_model(path=path)  # type:ignore[arg-type]
        kwargs.setdefault("stress_unit", "eV/A3")

        from matgl.ext.ase import PESCalculator as PESCalculator_

        return PESCalculator_(potential=model, **kwargs)

    @staticmethod
    def load_mtp(filename: str | Path, elements: list, **kwargs: Any) -> Calculator:
        """
        Load a machine-learned potential (MTPotential) from a configuration file and
        create a calculator object to interface with it.

        This method initializes an instance of MTPotential using a provided
        configuration file and elements. It returns a PESCalculator instance,
        which wraps the initialized potential model.

        Args:
            filename: MTP configuration file path.
            elements: Element symbols for the potential (e.g. ``["Cu"]``).
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the MTP model.
        """
        from maml.apps.pes import MTPotential

        model = MTPotential.from_config(filename=filename, elements=elements)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_gap(filename: str | Path, **kwargs: Any) -> Calculator:
        """
        Loads a Gaussian Approximation Potential (GAP) model from the given file and
        returns a corresponding Calculator instance. GAP is a machine learning-based
        potential used for atomistic simulations and requires a specific config file as
        input. Any additional arguments for the calculator can be passed via kwargs,
        allowing customization.

        Args:
            filename: GAP configuration file path.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the GAP model.
        """
        from maml.apps.pes import GAPotential

        model = GAPotential.from_config(filename=str(filename))
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_nnp(
        input_filename: str | Path,
        scaling_filename: str | Path,
        weights_filenames: list,
        **kwargs: Any,
    ) -> Calculator:
        """
        Loads a neural network potential (NNP) from specified configuration files and
        creates a Calculator object configured with the potential. This function allows
        for customizable keyword arguments to modify the behavior of the resulting
        Calculator.

        Args:
            input_filename: NNP input configuration path.
            scaling_filename: NNP scaling parameters path.
            weights_filenames: Paths to NNP weight files.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the NNP model.
        """
        from maml.apps.pes import NNPotential

        model = NNPotential.from_config(
            input_filename=input_filename,
            scaling_filename=scaling_filename,
            weights_filenames=weights_filenames,
        )
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_snap(param_file: str | Path, coeff_file: str | Path, **kwargs: Any) -> Calculator:
        """
        Load a SNAP (Spectral Neighbor Analysis Potential) configuration and create a
        corresponding Calculator instance.

        This static method initializes a SNAPotential instance using the provided
        configuration files and subsequently generates a PESCalculator based on the
        created potential model and additional keyword arguments.

        Args:
            param_file: SNAP parameter file path.
            coeff_file: SNAP coefficient file path.
            **kwargs: Forwarded to ``PESCalculator``.

        Returns:
            ``PESCalculator`` wrapping the SNAP model.
        """
        from maml.apps.pes import SNAPotential

        model = SNAPotential.from_config(param_file=param_file, coeff_file=coeff_file)
        return PESCalculator(potential=model, **kwargs)

    @staticmethod
    def load_ace(  # pragma: no cover
        basis_set: str | Path | ACEBBasisSet | ACECTildeBasisSet | BBasisConfiguration,
        **kwargs: Any,
    ) -> Calculator:
        """
        Load an ACE (Atomic Cluster Expansion) calculator using the specified basis set.

        This method utilizes the PyACE library to create and initialize a PyACECalculator
        instance with a given basis set. The provided basis set can take various forms including
        file paths, basis set objects, or configurations. Additional customization options
        can be passed through keyword arguments.

        Args:
            basis_set: ACE basis (path, or PyACE basis / configuration object).
            **kwargs: Forwarded to ``PyACECalculator``.

        Returns:
            Initialized PyACE ASE calculator.
        """
        from pyace import PyACECalculator

        return PyACECalculator(basis_set=basis_set, **kwargs)

    @staticmethod
    def load_nequip(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads and returns a NequIP `Calculator` instance from the specified model path.
        This method facilitates the integration of machine learning models into ASE
        by loading a model for atomic-scale simulations.

        Args:
            model_path: Path to the deployed NequIP model.
            **kwargs: Forwarded to ``NequIPCalculator.from_deployed_model``.

        Returns:
            NequIP ASE calculator instance.
        """
        from nequip.ase import NequIPCalculator

        return NequIPCalculator.from_deployed_model(model_path=model_path, **kwargs)

    @staticmethod
    def load_deepmd(  # pragma: no cover
        model_path: str | Path, **kwargs: Any
    ) -> Calculator:
        """
        Loads a Deep Potential Molecular Dynamics (DeePMD) model and returns a `Calculator`
        object for molecular dynamics simulations.

        This method imports the `deepmd.calculator.DP` class and initializes it with the
        given model path and optional keyword arguments. The resulting `Calculator` object
        is used to perform molecular simulations based on the specified DeePMD model.

        The function requires the DeePMD-kit library to be installed to properly import
        and utilize the `DP` class.

        Args:
            model_path: Trained DeePMD model path.
            **kwargs: Forwarded to DeePMD ``DP``.

        Returns:
            DeePMD ASE calculator instance.
        """
        from deepmd.calculator import DP

        return DP(model=model_path, **kwargs)

    @staticmethod
    def load_universal(name: str | Calculator, **kwargs: Any) -> Calculator:
        """
        Load a foundation potential calculator by its canonical name.

        Names follow the unified convention ``<Architecture>-<Dataset>[-<Functional>]-<Version>[-<Variant>]``
        (e.g. ``TensorNet-MatPES-PBE-2025.2``, ``MACE-MPA-0-medium``). The full list of
        canonical names is the keys of :data:`MODEL_REGISTRY`; short / legacy spellings
        in :data:`MODEL_ALIASES` resolve to a canonical name. Lookups are
        case-insensitive.

        If ``name`` is already a :class:`Calculator`, it is returned unchanged.

        Args:
            name: Canonical model name, alias, or an existing ASE calculator instance.
            **kwargs: Provider-specific options. These override the defaults stored
                in the registry entry (e.g. ``device="cuda"`` for ORB / FAIRChem).

        Returns:
            An ASE :class:`Calculator` instance.

        Raises:
            ValueError: If ``name`` is not a recognized model.
            ImportError: If the model is recognized but the provider's optional
                dependency is not installed.
        """
        if not isinstance(name, str):  # already an ASE Calculator
            return name

        canonical = _resolve_canonical(name)
        if canonical is None:
            # Backward-compat fallback: a raw MatGL pretrained model name passed
            # straight through (covers any newly released models not yet in the
            # registry) so users on the bleeding edge are not blocked.
            if name in _MATGL_AVAILABLE:
                return PESCalculator.load_matgl(name, **kwargs)
            suggestions = _suggest_models(name)
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            raise ValueError(
                f"Unrecognized {name=}.{hint} See matcalc.utils.MODEL_REGISTRY for the "
                f"full list of canonical names and matcalc.utils.MODEL_ALIASES for short "
                f"spellings."
            )

        spec = MODEL_REGISTRY[canonical]
        provider_kwargs = {k: v for k, v in spec.items() if k != "provider"}
        provider_kwargs.update(kwargs)  # user kwargs win over registry defaults
        provider = spec["provider"]

        loader = _PROVIDER_LOADERS.get(provider)
        if loader is None:
            raise ValueError(f"Unknown provider {provider!r} for model {canonical!r}.")
        try:
            return loader(provider_kwargs)
        except ImportError as e:
            raise ImportError(_install_hint(provider)) from e


# --- Per-provider loaders for ``PESCalculator.load_universal`` ---------------
#
# Each loader takes a ``kwargs`` dict (already merged from the registry entry
# and the user's overrides) and returns an ASE ``Calculator``. ImportErrors
# raised inside a loader are translated by ``load_universal`` into a friendly
# "pip install matcalc[<extra>]" message via ``_install_hint``.


def _load_matgl(kwargs: dict[str, Any]) -> Calculator:
    path = kwargs.pop("path")
    return PESCalculator.load_matgl(path, **kwargs)


def _load_mace_mp(kwargs: dict[str, Any]) -> Calculator:
    from mace.calculators import mace_mp

    return mace_mp(**kwargs)


def _load_sevennet(kwargs: dict[str, Any]) -> Calculator:
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(**kwargs)


def _load_grace(kwargs: dict[str, Any]) -> Calculator:
    from tensorpotential.calculator.foundation_models import grace_fm

    return grace_fm(**kwargs)


def _load_orb(kwargs: dict[str, Any]) -> Calculator:
    from orb_models.forcefield.inference.calculator import ORBCalculator
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

    model = kwargs.pop("model")
    device = kwargs.get("device", "cpu")
    orbff, atoms_adapter = ORB_PRETRAINED_MODELS[model](device=device)
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, **kwargs)


def _load_mattersim(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(**kwargs)


def _load_fairchem(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    device = kwargs.pop("device", "cpu")
    model = kwargs.pop("model")
    task_name = kwargs.pop("task_name")
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task_name, **kwargs)


def _load_upet(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    from upet.ase import UPETCalculator

    return UPETCalculator(**kwargs)


def _load_deepmd(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    from deepmd.calculator import DP

    return DP(**kwargs)


_PROVIDER_LOADERS: dict[str, Callable[[dict[str, Any]], Calculator]] = {
    "matgl": _load_matgl,
    "mace_mp": _load_mace_mp,
    "sevennet": _load_sevennet,
    "grace": _load_grace,
    "orb": _load_orb,
    "mattersim": _load_mattersim,
    "fairchem": _load_fairchem,
    "upet": _load_upet,
    "deepmd": _load_deepmd,
}


def to_ase_atoms(structure: Atoms | Structure | Molecule) -> Atoms:
    """
    Converts a given structure into an ASE Atoms object. If the input is already
    an ``Atoms``, a shallow copy is returned so that downstream code (which
    routinely attaches a calculator and runs optimizers in-place) cannot mutate
    a caller-owned object — important for parallel ``calc_many`` execution
    where joblib workers must not share state.

    Args:
        structure: ASE ``Atoms``, pymatgen ``Structure``, or ``Molecule``.

    Returns:
        Fresh ASE ``Atoms`` for the same system.
    """
    return structure.copy() if isinstance(structure, Atoms) else AseAtomsAdaptor.get_atoms(structure)


def to_pmg_structure(structure: Atoms | Structure) -> Structure:
    """
    Converts a given structure of type Atoms or Structure into a Structure
    object. If the input structure is already of type Structure, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Structure using the AseAtomsAdaptor.

    Args:
        structure: ASE ``Atoms`` or pymatgen ``Structure``.

    Returns:
        Pymatgen ``Structure`` (unchanged if already a structure).
    """
    return structure if isinstance(structure, Structure) else AseAtomsAdaptor.get_structure(structure)  # type: ignore[return-value]


def to_pmg_molecule(structure: Atoms | Structure | Molecule | IMolecule) -> IMolecule:
    """
    Converts a given structure of type Atoms or Structure into a Molecule
    object. If the input structure is already of type Molecule, it is
    returned unchanged. If the input structure is of type Atoms, it is
    converted to a Molecule using the AseAtomsAdaptor.

    Args:
        structure: ASE ``Atoms``, pymatgen ``Structure`` / ``Molecule``, or interface molecule type.

    Returns:
        Pymatgen ``Molecule`` representation.
    """
    if isinstance(structure, Atoms):
        structure = AseAtomsAdaptor.get_molecule(structure)

    return Molecule.from_sites(structure)  # type: ignore[return-value]
