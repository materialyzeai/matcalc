"""Some utility methods, e.g., for getting calculators from well-known sources."""

from __future__ import annotations

import difflib
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .units import eVA3ToGPa

if TYPE_CHECKING:
    from collections.abc import Callable

    from maml.apps.pes import LMPStaticCalculator
    from pyace.basis import ACEBBasisSet, ACECTildeBasisSet, BBasisConfiguration
    from pymatgen.core import IMolecule, IStructure


# Unified naming convention for foundation potentials:
#
#     <Architecture>-<Dataset>-<Optional Version>
#
# e.g. ``TensorNet-MatPES-PBE-2025.2`` or ``MACE-MPA-0-medium``. Each entry in
# ``MODEL_REGISTRY`` maps a canonical name to a provider and the provider-specific
# kwargs used to materialise the calculator. Users select a model by its canonical
# name; ``MODEL_ALIASES`` provides short / legacy spellings that resolve to a
# canonical name.
#
# To add a new model, append an entry here. The ``provider`` key picks the loader
# branch in ``PESCalculator.load_universal``; remaining keys are forwarded to that
# loader (user kwargs to ``load_universal`` override these defaults).
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # MatGL — TensorNet on MatPES
    "TensorNet-MatPES-PBE-2025.2": {"provider": "matgl", "path": "TensorNet-PES-MatPES-PBE-2025.2"},
    "TensorNet-MatPES-r2SCAN-2025.2": {"provider": "matgl", "path": "TensorNet-PES-MatPES-r2SCAN-2025.2"},
    # MatGL — M3GNet
    "M3GNet-MatPES-PBE-2025.1": {"provider": "matgl", "path": "M3GNet-PES-MatPES-PBE-2025.1"},
    # MatGL — CHGNet
    "CHGNet-MatPES-PBE-2025.2.10": {"provider": "matgl", "path": "CHGNet-PES-MatPES-PBE-2025.2.10"},
    "CHGNet-MatPES-r2SCAN-2025.2.10": {"provider": "matgl", "path": "CHGNet-PES-MatPES-r2SCAN-2025.2.10"},
    "CHGNet-MPtrj-2023.12.1-2.7M": {"provider": "matgl", "path": "CHGNet-MPtrj-2023.12.1-2.7M-PES"},
    "CHGNet-MPtrj-2024.2.13-11M": {"provider": "matgl", "path": "CHGNet-MPtrj-2024.2.13-11M-PES"},
    # MACE foundation models (mace-foundations release names)
    "MACE-MP-0-small": {"provider": "mace_mp", "model": "small"},
    "MACE-MP-0-medium": {"provider": "mace_mp", "model": "medium"},
    "MACE-MP-0-large": {"provider": "mace_mp", "model": "large"},
    "MACE-MP-0b-small": {"provider": "mace_mp", "model": "small-0b"},
    "MACE-MP-0b-medium": {"provider": "mace_mp", "model": "medium-0b"},
    "MACE-MP-0b2-small": {"provider": "mace_mp", "model": "small-0b2"},
    "MACE-MP-0b2-medium": {"provider": "mace_mp", "model": "medium-0b2"},
    "MACE-MP-0b2-large": {"provider": "mace_mp", "model": "large-0b2"},
    "MACE-MP-0b3-medium": {"provider": "mace_mp", "model": "medium-0b3"},
    "MACE-MPA-0-medium": {"provider": "mace_mp", "model": "medium-mpa-0"},
    "MACE-OMAT-0-small": {"provider": "mace_mp", "model": "small-omat-0"},
    "MACE-OMAT-0-medium": {"provider": "mace_mp", "model": "medium-omat-0"},
    "MACE-MatPES-PBE-0": {"provider": "mace_mp", "model": "mace-matpes-pbe-0"},
    "MACE-MatPES-r2SCAN-0": {"provider": "mace_mp", "model": "mace-matpes-r2scan-0"},
    # NOTE: MACE multi-head checkpoints (mh-0, mh-1) require a mandatory ``head=...``
    # kwarg per use; they are not registered as default-loadable canonical names.
    # SevenNet (model names match the upstream HF / SevenNetCalculator strings)
    "SevenNet-0": {"provider": "sevennet", "model": "7net-0"},
    "SevenNet-l3i5": {"provider": "sevennet", "model": "7net-l3i5"},
    "SevenNet-MF-OMPA": {"provider": "sevennet", "model": "7net-mf-ompa"},
    "SevenNet-OMAT": {"provider": "sevennet", "model": "7net-omat"},
    # GRACE / TensorPotential
    "GRACE-1L-OAM": {"provider": "grace", "model": "GRACE-1L-OAM"},
    "GRACE-2L-OAM": {"provider": "grace", "model": "GRACE-2L-OAM"},
    "GRACE-2L-OMAT": {"provider": "grace", "model": "GRACE-2L-OMAT"},
    "GRACE-2L-MPtrj": {"provider": "grace", "model": "GRACE-2L-MPtrj"},
    # Orb
    "ORB-v2": {"provider": "orb", "model": "orb-v2"},
    "ORB-d3-v2": {"provider": "orb", "model": "orb-d3-v2"},
    "ORB-d3-sm-v2": {"provider": "orb", "model": "orb-d3-sm-v2"},
    "ORB-d3-xs-v2": {"provider": "orb", "model": "orb-d3-xs-v2"},
    # MatterSim
    "MatterSim-v1.0.0-1M": {"provider": "mattersim", "load_path": "MatterSim-v1.0.0-1M.pth"},
    "MatterSim-v1.0.0-5M": {"provider": "mattersim", "load_path": "MatterSim-v1.0.0-5M.pth"},
    # FAIRChem (UMA family) — upstream uses ``uma-s-1p2`` style; we expose them
    # under the unified ``<Arch>-<Size>-<Version>`` form.
    "UMA-S-1.2": {"provider": "fairchem", "model": "uma-s-1p2", "task_name": "omat"},
    "UMA-S-1.1": {"provider": "fairchem", "model": "uma-s-1p1", "task_name": "omat"},
    "UMA-M-1.1": {"provider": "fairchem", "model": "uma-m-1p1", "task_name": "omat"},
    # PET-MAD
    "PETMAD-1.0.0": {"provider": "petmad"},
    # DeePMD-LAM
    "DPA3-LAM-2025.3.14": {"provider": "deepmd"},
}

# Short / legacy aliases. Keys are matched case-insensitively. Values must be
# canonical names from ``MODEL_REGISTRY``.
MODEL_ALIASES: dict[str, str] = {
    # short architecture / functional aliases — pick a sensible default per family
    "tensornet": "TensorNet-MatPES-PBE-2025.2",
    "m3gnet": "M3GNet-MatPES-PBE-2025.1",
    "chgnet": "CHGNet-MatPES-PBE-2025.2.10",
    "pbe": "TensorNet-MatPES-PBE-2025.2",
    "r2scan": "TensorNet-MatPES-r2SCAN-2025.2",
    "mace": "MACE-MPA-0-medium",
    "sevennet": "SevenNet-0",
    "grace": "GRACE-2L-OAM",
    "tensorpotential": "GRACE-2L-OAM",
    "orb": "ORB-v2",
    "mattersim": "MatterSim-v1.0.0-1M",
    "fairchem": "UMA-S-1.2",
    "uma": "UMA-S-1.2",
    "petmad": "PETMAD-1.0.0",
    "deepmd": "DPA3-LAM-2025.3.14",
    # legacy MatGL spellings — keep working so existing code/notebooks don't break
    "tensornet-pes-matpes-pbe-2025.2": "TensorNet-MatPES-PBE-2025.2",
    "tensornet-pes-matpes-r2scan-2025.2": "TensorNet-MatPES-r2SCAN-2025.2",
    "m3gnet-pes-matpes-pbe-2025.1": "M3GNet-MatPES-PBE-2025.1",
    "chgnet-pes-matpes-pbe-2025.2.10": "CHGNet-MatPES-PBE-2025.2.10",
    "chgnet-pes-matpes-r2scan-2025.2.10": "CHGNet-MatPES-r2SCAN-2025.2.10",
    "chgnet-matpes-pbe-2025.2.10-2.7m-pes": "CHGNet-MatPES-PBE-2025.2.10",
    "chgnet-mptrj-2023.12.1-2.7m-pes": "CHGNet-MPtrj-2023.12.1-2.7M",
    "chgnet-mptrj-2024.2.13-11m-pes": "CHGNet-MPtrj-2024.2.13-11M",
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
    "petmad": "petmad",
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

        Names follow the unified convention ``<Architecture>-<Dataset>-<Optional Version>``
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
    from orb_models.forcefield.calculator import ORBCalculator
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS

    model = kwargs.pop("model")
    device = kwargs.get("device", "cpu")
    orbff = ORB_PRETRAINED_MODELS[model](device=device)
    return ORBCalculator(orbff, **kwargs)


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


def _load_petmad(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    from pet_mad.calculator import PETMADCalculator

    return PETMADCalculator(**kwargs)


# Bundled DPA3 checkpoint under tests/ (only present in editable installs).
# When matcalc is installed from a wheel the user must pass ``model=<path>``
# explicitly.
_DPA3_BUNDLED_PATH = (
    Path(__file__).parent.absolute() / "../../tests/pes/DPA3-LAM-2025.3.14-PES" / "2025-03-14-dpa3-openlam.pth"
).resolve()


def _load_deepmd(kwargs: dict[str, Any]) -> Calculator:  # pragma: no cover
    if "model" not in kwargs:
        if _DPA3_BUNDLED_PATH.exists():
            kwargs["model"] = _DPA3_BUNDLED_PATH
        else:
            raise FileNotFoundError(
                "DeePMD model path not found. Pass model=<path-to-.pth> explicitly to "
                "load_universal(..., model=...), or download the DPA3-LAM checkpoint and "
                "supply its path. (Bundled fallback is only available from an editable "
                "checkout.)"
            )
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
    "petmad": _load_petmad,
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
