"""This module implements classes for running benchmarks on materials properties."""

from __future__ import annotations

import abc
import gzip
import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from scipy.optimize import curve_fit
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator

    from ._base import PropCalc

from ._elasticity import ElasticityCalc
from ._phonon import PhononCalc
from ._relaxation import RelaxCalc
from ._stability import EnergeticsCalc
from .backend import run_pes_calc
from .config import BENCHMARK_DATA_DIR, BENCHMARK_HF_REPO_ID

logger = logging.getLogger(__name__)


def get_available_benchmarks() -> list[str]:
    """
    Fetches and returns a list of available benchmarks from the
    ``materialyze/matcalc-bench`` Hugging Face dataset.

    Returns:
        Benchmark archive filenames ending in ``.json.gz``.
    """
    files = HfApi().list_repo_files(BENCHMARK_HF_REPO_ID, repo_type="dataset")
    return [f for f in files if f.endswith(".json.gz")]


def get_benchmark_data(name: str) -> list[Any]:
    """
    Retrieve a benchmark dataset from the ``materialyze/matcalc-bench`` Hugging
    Face dataset. Files are cached locally by ``huggingface_hub``.

    Args:
        name: Benchmark JSON archive filename (e.g. ``*.json.gz``).

    Returns:
        List of entries decoded with ``MontyDecoder`` (typically dicts).

    Raises:
        huggingface_hub.errors.EntryNotFoundError: If the file does not exist
            in the dataset.
    """
    local_path = hf_hub_download(
        repo_id=BENCHMARK_HF_REPO_ID,
        filename=name,
        repo_type="dataset",
        cache_dir=str(BENCHMARK_DATA_DIR),
    )
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        return json.load(f, cls=MontyDecoder)


class CheckpointFile:
    """
    Represents a checkpoint file system management utility.

    This class provides mechanisms to manage and process a file path and its
    associated actions such as loading and saving data. It ensures standardized
    path handling through the use of `Path` objects, enables loading checkpoint
    data from a file, and facilitates the saving of resulting data.

    Attributes:
        path: Checkpoint file path as ``pathlib.Path``.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        """
        Represents an initialization process for handling a filesystem path. The
        provided path is converted into a `Path` object for standardized path
        management in the application.

        Args:
            path: Filesystem path as str or ``Path``.
        """
        self.path = Path(path)

    def load(self, *args: list) -> tuple:
        """
        Loads checkpoint data from a specified path if it exists, returning the loaded entries along with
        remaining portions of the given input arguments.

        The method checks if the file path exists, and if so, it loads data from the specified
        file using a predefined `loadfn` function. It logs the number of loaded entries and
        returns the successfully loaded entries alongside sliced input arguments based on the
        number of loaded entries. If the file path does not exist, it returns empty results
        and the original input arguments unchanged.

        Args:
            *args: Additional list arguments aligned with checkpoint rows (sliced after load).

        Returns:
            Tuple ``(loaded_rows, *tail_slices)``; ``loaded_rows`` is empty if missing file.
        """
        if self.path.exists():
            results = loadfn(self.path)
            logger.info("Loaded %d entries from %s...", len(results), self.path)
            return results, *[a[len(results) :] for a in args]
        return [], *args

    def save(self, results: list[dict[str, Any]]) -> None:
        """
        Saves a list of results at the specified checkpoint location.

        Args:
            results: Rows to serialize to the checkpoint path.
        """
        logger.info("Saving %d entries to %s...", len(results), self.path)
        dumpfn(results, self.path)


class Benchmark(metaclass=abc.ABCMeta):
    """
    Abstract base for property benchmarks against published datasets.

    Loads benchmark JSON, builds per-row ground-truth records and structure list, and runs a
    ``PropCalc`` over structures (with optional checkpointing).

    Attributes:
        properties: Target property keys to compare.
        other_fields: Extra benchmark fields copied into each row.
        index_name: Primary key column (e.g. material id).
        structures: Pymatgen structures in benchmark order.
        kwargs: Kwargs forwarded to ``get_prop_calc``.
        property_rename_map: Optional column renames on the results frame.
        ground_truth: List of dict rows (converted to DataFrame in ``run``).
    """

    def __init__(
        self,
        benchmark_name: str | Path,
        properties: Sequence[str],
        index_name: str,
        other_fields: tuple = (),
        property_rename_map: dict[str, str] | None = None,
        suffix_ground_truth: str = "DFT",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Args:
            benchmark_name: Remote benchmark filename or local path to JSON.
            properties: Keys to pull from each benchmark entry as targets.
            index_name: Row id field name.
            other_fields: Extra fields to copy into each ground-truth row.
            property_rename_map: Renames applied to result columns.
            suffix_ground_truth: Suffix for reference columns (e.g. ``_DFT``).
            n_samples: Random subsample size, or None for all.
            seed: RNG seed when subsampling.
            **kwargs: Forwarded to ``get_prop_calc``.

        Raises:
            FileNotFoundError: If ``benchmark_name`` is a missing local path.
            ValueError: If entries lack required fields.
        """
        rows = []
        structures = []
        entries = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if n_samples:
            random.seed(seed)
            entries = random.sample(entries, n_samples)

        # We will first create a DataFrame from the required components from the raw benchmark entries.
        # We also create the list of structures in the order of the entries.
        for entry in entries:
            row = {k: entry[k] for k in [index_name, *other_fields]}
            for prop in properties:
                row[f"{prop}_{suffix_ground_truth}"] = entry[prop]
            rows.append(row)

            structures.append(entry["structure"])

        self.properties = properties
        self.other_fields = other_fields
        self.index_name = index_name
        self.structures = structures
        self.kwargs = kwargs
        self.property_rename_map = property_rename_map or {}
        self.ground_truth = rows

    @abc.abstractmethod
    def get_prop_calc(self, calculator: str | Calculator, **kwargs: Any) -> PropCalc:
        """
        Abstract method to retrieve a property calculation object using the provided calculator and additional
        parameters.
        This method must be implemented by subclasses and will utilize the provided calculator to create a
        `PropCalc` instance, possibly influenced by additional keyword arguments.

        Args:
            calculator: ASE calculator or model name string.
            **kwargs: Merged with ``self.kwargs`` in ``run`` (subclass dependent).

        Returns:
            Configured ``PropCalc`` instance.
        """

    @abc.abstractmethod
    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Implements post-processing of results. A default implementation is provided that simply appends the model name
        as a suffix to the key of the input dictionary for all properties. Subclasses can override this method to
        provide more sophisticated processing.

        Args:
            result: Output dict from ``PropCalc.calc``, or None on failure.
            model_name: Suffix tag for predicted columns.

        Returns:
            Flat dict mapping ``{prop}_{model_name}`` to values (or None).
        """
        return (
            {f"{k}_{model_name}": result[k] for k in self.properties}
            if result is not None
            else {f"{k}_{model_name}": None for k in self.properties}
        )

    def run(
        self,
        calculator: str | Calculator,
        model_name: str,
        *,
        n_jobs: None | int = -1,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 1000,
        delete_checkpoint_on_finish: bool = True,
        include_full_results: bool = False,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Processes a collection of structures using a calculator, saves intermittent
        checkpoints, and returns the results in a DataFrame. This function supports
        parallel computation and allows for error tolerance during processing.

        The function also retrieves a property calculator and utilizes it to calculate
        desired results for the given set of structures. Checkpoints are saved
        periodically based on the specified frequency, ensuring that progress is not
        lost in case of interruptions.

        Args:
            calculator: ASE calculator or universal model name.
            model_name: Label appended to predicted property columns.
            n_jobs: ``joblib`` parallelism for ``calc_many`` (-1 = all cores).
            checkpoint_file: Optional path to resume/save partial results.
            checkpoint_freq: Save checkpoint every this many completed structures.
            delete_checkpoint_on_finish: Remove checkpoint file after success.
            include_full_results: Keep all ``calc`` keys, not only ``properties``.
            **kwargs: Forwarded to ``calc_many`` / calculator.

        Returns:
            DataFrame of ground-truth rows plus model predictions.
        """
        checkpoint = None
        if checkpoint_file:
            checkpoint = CheckpointFile(checkpoint_file)
            results, ground_truth, structures = checkpoint.load(self.ground_truth, self.structures)
        else:
            results = []
            ground_truth = self.ground_truth
            structures = list(self.structures)

        prop_calc = self.get_prop_calc(calculator, **self.kwargs)
        # We make use of the generator from prop_calc.calc_many to do this in a memory efficient manner.
        # allow_errors typically should be true since some of the calculations may fail.
        for row, d in tqdm(
            zip(ground_truth, prop_calc.calc_many(structures, n_jobs=n_jobs, allow_errors=True, **kwargs), strict=True),
            total=len(structures),
        ):
            r = dict(row)
            r.update(self.process_result(d, model_name))
            if include_full_results and d is not None:
                r.update({k: v for k, v in d.items() if k not in self.properties})

            results.append(r)
            if checkpoint and len(results) % checkpoint_freq == 0:
                checkpoint.save(results)

        if delete_checkpoint_on_finish and checkpoint_file:
            try:
                os.remove(checkpoint_file)
            except FileNotFoundError:
                pass

        results_df = pd.DataFrame(results)
        if self.property_rename_map:

            def _rename_property(col: str) -> str:
                for k, v in self.property_rename_map.items():
                    col = col.replace(k, v)
                return col

            results_df = results_df.rename(columns=_rename_property)
        return results_df


class EquilibriumBenchmark(Benchmark):
    """
    Represents a benchmark for evaluating and analyzing equilibrium properties of materials.
    This benchmark utilizes a dataset and provides functionality for property calculation
    and result processing. The class is designed to work with a predefined framework for
    benchmarking equilibrium properties. The benchmark dataset contains data such as relaxed
    structures, un-/corrected formation energy along with additional metadata. This class
    supports configurability through metadata files, index names, and additional benchmark
    properties. It relies on external calculators and utility classes for property computations
    and result handling.
    """

    def __init__(
        self,
        index_name: str = "material_id",
        benchmark_name: str | Path = "wbm-random-pbe52-equilibrium-2025.1.json.gz",
        folder_name: str = "default_folder",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the EquilibriumBenchmark instance with specified benchmark metadata and
        configuration parameters. It sets up the benchmark with the necessary properties
        required for equilibrium benchmark analysis.

        Args:
            index_name: Primary key field in benchmark rows.
            benchmark_name: Remote filename or local path to benchmark JSON.
            folder_name: Label for file/artifact grouping.
            **kwargs: Forwarded to ``Benchmark`` (properties, sampling, etc.).
        """
        self.folder_name = folder_name
        kwargs.setdefault("properties", ("structure", "formation_energy_per_atom"))
        kwargs.setdefault("property_rename_map", {"formation_energy_per_atom": "Eform"})
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(benchmark_name, index_name=index_name, **kwargs)

    def get_prop_calc(self, calculator: str | Calculator, **kwargs: Any) -> PropCalc:
        """
        Returns a property calculation object for performing relaxation and formation energy
        calculations. This method initializes the stability calculator using the provided
        Calculator object and any additional configuration parameters.

        Args:
            calculator: ASE calculator or model name.
            **kwargs: Merged into ``EnergeticsCalc`` (after elemental ref prep).

        Returns:
            ``EnergeticsCalc`` with MP-PBE elemental references relaxed per element.
        """
        self._prepare_elemental_refs(calculator)
        return EnergeticsCalc(
            calculator,
            elemental_refs=self.elemental_refs,
            use_gs_reference=True,
            fmax=0.05,
            perturb_distance=0.1,
            **kwargs,
        )

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary containing final structures and formation energy per atom,
        formats the keys according to the provided model name. If the result is None, default values
        of NaN are returned for final structures or formation energy per atom.

        Args:
            result: ``EnergeticsCalc`` output dict, or None if calculation failed.
            model_name: Suffix for column keys.

        Returns:
            Dict with ``structure_{model_name}`` and ``formation_energy_per_atom_{model_name}``.
        """
        return {
            f"structure_{model_name}": (result["final_structure"] if result is not None else float("nan")),
            f"formation_energy_per_atom_{model_name}": (
                result["formation_energy_per_atom"] if result is not None else float("nan")
            ),
        }

    def run(
        self,
        calculator: str | Calculator,
        model_name: str,
        *,
        n_jobs: None | int = -1,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 1000,
        delete_checkpoint_on_finish: bool = True,
        include_full_results: bool = False,
        **kwargs,  # noqa:ANN003
    ) -> pd.DataFrame:
        """
        Processes a collection of structures using a calculator, saves intermittent checkpoints,
        and returns the results in a DataFrame. In addition to the base processing performed
        by the parent class, this method computes the Euclidean distance between the relaxed
        structure (obtained from the property calculation) and the reference DFT structure,
        using SiteStatsFingerprint. The computed distance is added as a new column in the
        results DataFrame with the key "distance_{model_name}".

        This function supports parallel computation and allows for error tolerance during processing.
        It retrieves a property calculator and utilizes it to calculate desired results for the given
        set of structures. Checkpoints are saved periodically based on the specified frequency,
        ensuring that progress is not lost in case of interruptions.

        Args:
            calculator: ASE calculator or model name.
            model_name: Label for predicted columns.
            n_jobs: Parallelism for ``calc_many``.
            checkpoint_file: Optional resume path.
            checkpoint_freq: Checkpoint interval (structures).
            delete_checkpoint_on_finish: Remove checkpoint after success.
            include_full_results: Keep full ``calc`` dict keys.
            **kwargs: Forwarded to ``Benchmark.run``.

        Returns:
            Results frame plus ``d_{model_name}`` structure fingerprint distance to DFT.
        """
        results_df = super().run(
            calculator,
            model_name,
            n_jobs=n_jobs,
            checkpoint_file=checkpoint_file,
            checkpoint_freq=checkpoint_freq,
            delete_checkpoint_on_finish=delete_checkpoint_on_finish,
            include_full_results=include_full_results,
            **kwargs,
        )

        ssf = SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
            stats=("mean", "std_dev", "minimum", "maximum"),
        )

        results_df[f"d_{model_name}"] = [
            np.linalg.norm(np.array(ssf.featurize(model)) - np.array(ssf.featurize(dft)))
            if model is not None and dft is not None
            else np.nan
            for model, dft in zip(results_df[f"structure_{model_name}"], results_df["structure_DFT"], strict=True)
        ]

        return results_df

    def _prepare_elemental_refs(self, calculator: str | Calculator) -> None:
        """
        Helper function to prepare and cache ground-state reference energies for all elements in the benchmark.

        This method performs the following steps exactly once per Benchmark instance:
        1. Load the full elemental references.
        2. Traverse `self.structures` to collect the set of unique element symbols needed.
        3. For each symbol:
            a. Retrieve its reference structure(s) from the full dataset.
            b. Use RelaxCalc to relax each structure and calculate the energy.
            c. Screen out the minimum energy per atom among those structures.
        4. Populate `self.elemental_refs` as a dict mapping each elemental symbol.

        After this run, subsequent calls into EnergeticsCalc with use_gs_reference=True
        will simply look up values in `self.elemental_refs`, avoiding repeated relaxations.

        Args:
            calculator: ASE calculator or universal model name for elemental relaxations.
        """
        full_refs = loadfn(Path(__file__).parent / "elemental_refs" / "MP-PBE-Element-Refs.json.gz")
        needed_els = {el.symbol for struct in self.structures for el in struct.composition.elements}
        relaxer = RelaxCalc(calculator, fmax=0.05)
        self.elemental_refs = {}
        for el in needed_els:
            info = full_refs[el]
            struct_list = [info["structure"]] if not isinstance(info["structure"], list) else info["structure"]
            energy_per_atom = [
                r["energy"] / r["final_structure"].num_sites for r in relaxer.calc_many(struct_list) if r is not None
            ]
            self.elemental_refs[el] = {
                "energy_per_atom": min(energy_per_atom),
            }


class ElasticityBenchmark(Benchmark):
    """
    Represents a benchmark for evaluating and analyzing mechanical properties such as
    bulk modulus and shear modulus for various materials. The benchmark primarily utilizes
    a dataset and provides functionality for property calculation and result processing.

    The class is designed to work with a predefined framework for benchmarking mechanical
    properties. The benchmark dataset contains values such as bulk modulus and shear
    modulus along with additional metadata. This class supports configurability through
    metadata files, index names, and additional benchmark properties. It relies on
    external calculators and utility classes for property computations and result handling.
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "mp-binary-pbe-elasticity-2025.1.json.gz",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes the ElasticityBenchmark instance by taking benchmark metadata and
        additional configuration parameters. Sets up the benchmark framework with
        specified mechanical properties and metadata.

        Args:
            index_name: Primary key field name.
            benchmark_name: Remote filename or local benchmark path.
            **kwargs: Forwarded to ``Benchmark``.
        """
        kwargs.setdefault("properties", ("bulk_modulus_vrh", "shear_modulus_vrh"))
        kwargs.setdefault("property_rename_map", {"bulk_modulus": "K", "shear_modulus": "G"})
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(
            benchmark_name,
            index_name=index_name,
            **kwargs,
        )

    def get_prop_calc(self, calculator: str | Calculator, **kwargs: Any) -> PropCalc:
        """
        Calculates and returns a property calculation object based on the provided
        calculator and optional parameters. This is useful for initializing and
        configuring a property calculation.

        Args:
            calculator: ASE calculator or model name.
            **kwargs: Merged into ``ElasticityCalc`` (default ``fmax`` 0.05).

        Returns:
            Configured ``ElasticityCalc``.
        """
        kwargs.setdefault("fmax", 0.05)
        kwargs.setdefault("units_GPa", True)
        return ElasticityCalc(calculator, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary containing bulk and shear modulus values, adjusts
        them by multiplying with a predefined conversion factor, and formats the keys
        according to the provided model name. If the result is None, default values of
        NaN are returned for both bulk and shear modulus.

        Args:
            result: ``ElasticityCalc`` output or None.
            model_name: Column suffix for predictions.

        Returns:
            K and G in GPa as ``bulk_modulus_vrh_{model}``, ``shear_modulus_vrh_{model}``.
        """
        return {
            f"bulk_modulus_vrh_{model_name}": (result["bulk_modulus_vrh"] if result is not None else float("nan")),
            f"shear_modulus_vrh_{model_name}": (result["shear_modulus_vrh"] if result is not None else float("nan")),
        }


class PhononBenchmark(Benchmark):
    """
    Phonon benchmark: compares heat capacity at 300 K (configurable index) to reference data.

    Wraps ``PhononCalc`` with benchmark-specific defaults and result extraction.
    """

    def __init__(
        self,
        index_name: str = "mp_id",
        benchmark_name: str | Path = "alexandria-binary-pbe-phonon-2025.1.json.gz",
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Initializes an instance with specified index and benchmark details.

        This constructor sets up an object with predefined properties such as heat
        capacity and additional fields such as the formula. It supports customizations
        via keyword arguments for further configurations.

        Args:
            index_name: Primary key field name.
            benchmark_name: Remote filename or local path.
            **kwargs: Forwarded to ``Benchmark``.
        """
        kwargs.setdefault("properties", ("heat_capacity",))
        kwargs.setdefault("property_rename_map", {"heat_capacity": "CV"})
        kwargs.setdefault("other_fields", ("formula",))
        super().__init__(
            benchmark_name,
            index_name=index_name,
            **kwargs,
        )

    def get_prop_calc(self, calculator: str | Calculator, **kwargs: Any) -> PropCalc:
        """
        Retrieves a phonon calculation instance based on the given calculator and
        additional keyword arguments.

        This function initializes and returns a `PhononCalc` object using the provided
        calculator instance and any optional keyword arguments to configure the
        calculation further.

        Args:
            calculator: ASE calculator or model name.
            **kwargs: Merged into ``PhononCalc`` (defaults ``fmax=0.05``, no phonon YAML).

        Returns:
            ``PhononCalc`` instance.
        """
        return PhononCalc(calculator, fmax=0.05, write_phonon=False, **kwargs)

    def process_result(self, result: dict | None, model_name: str) -> dict:
        """
        Processes the result dictionary to extract specific thermal property information for the provided model name.

        Args:
            result: ``PhononCalc`` output dict or None.
            model_name: Column suffix.

        Returns:
            ``heat_capacity_{model_name}`` at fixed thermal-properties index (default grid).
        """
        return {
            f"heat_capacity_{model_name}": (
                result["thermal_properties"]["heat_capacity"][30] if result is not None else float("nan")
            )
        }


class SofteningBenchmark:
    """
    A benchmark for the systematic softening of a PES, as described in:
        B. Deng, et al. npj Comput. Mater. 11, 9 (2025).
        doi: 10.1038/s41524-024-01500-6
    The dataset used here can be found in figshare through:
        https://figshare.com/articles/dataset/WBM_high_energy_states/27307776?file=50005317
    This benchmark essentially performs static calculation on pre-sampled high-energy
    PES configurations, and then compare the systematic underestimation of forces
    predicted between GGA-DFT and the provided force field.
    """

    def __init__(
        self,
        benchmark_name: str | Path = "wbm-high-energy-states.json.gz",
        index_name: str = "wbm_id",
        n_samples: int | None = None,
        seed: int = 42,
        **kwargs,  # noqa:ANN003
    ) -> None:
        """
        Args:
            benchmark_name: Remote filename or local path to high-energy-state dataset.
            index_name: Id field name for each material block.
            n_samples: Optional random subset of material keys.
            seed: RNG seed for subsampling.
            **kwargs: Stored on ``self.kwargs`` for extensions.
        """
        self.index_name = index_name
        raw: Any = get_benchmark_data(benchmark_name) if isinstance(benchmark_name, str) else loadfn(benchmark_name)
        if not isinstance(raw, dict):
            raise TypeError(
                "SofteningBenchmark expects a dict dataset (material_id -> frames); "
                f"got {type(raw).__name__} from {benchmark_name!r}."
            )
        self.data: dict[str, Any] = raw
        if n_samples:
            random.seed(seed)
            self.material_ids = random.sample(list(self.data.keys()), n_samples)
        else:
            self.material_ids = list(self.data.keys())
        self.kwargs = kwargs

    @staticmethod
    def get_linear_fitted_slope(x: list | np.ndarray, y: list | np.ndarray) -> float:
        """
        Linear least-squares slope for ``y ≈ a x`` (proportional fit).

        Args:
            x: Reference force components (flattened).
            y: Predicted force components (flattened).

        Returns:
            Fitted proportionality constant ``a``.
        """
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        def linear_model(x: float, a: float) -> float:
            return a * x

        popt, _ = curve_fit(linear_model, x, y)
        return popt[0]

    def run(
        self,
        calculator: Calculator,
        model_name: str,
        checkpoint_file: str | Path | None = None,
        checkpoint_freq: int = 10,
        *,
        include_full_results: bool = False,
    ) -> pd.DataFrame:
        """
        Process all the material ids by
        1. calculate the forces on all the sampled structures.
        2. perform a linear fit on the predicted forces w.r.t. provided DFT forces.
        3. returning the fitted slopes as the softening scales.

        Args:
            calculator: ASE calculator for force evaluation.
            model_name: Column suffix for softening scale.
            checkpoint_file: Optional resume path.
            checkpoint_freq: Checkpoint every N materials completed.
            include_full_results: Keep ``raw_force_predictions`` column when True.

        Returns:
            DataFrame with per-material softening scale and metadata.
        """
        checkpoint = None
        results = []
        material_ids = self.material_ids
        if checkpoint_file:
            checkpoint = CheckpointFile(checkpoint_file)
            results, material_ids = checkpoint.load(self.material_ids)

        for material_id in material_ids:
            frames = self.data[material_id]
            test_structure = next(iter(frames.values()))["structure"]

            try:
                force_ground_truth, force_prediction = [], []
                for frame in frames.values():
                    r = run_pes_calc(frame["structure"], calculator)
                    forces = r.forces.tolist()
                    force_prediction.append(forces)
                    force_ground_truth.append(frame["vasp_f"])

                softening_scale = self.get_linear_fitted_slope(force_ground_truth, force_prediction)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SofteningBenchmark: failed for material_id=%s (%s): %s",
                    material_id,
                    type(exc).__name__,
                    exc,
                )
                softening_scale = None
                force_prediction = None

            results.append(
                {
                    "material_id": material_id,
                    "formula": test_structure.composition.reduced_formula,
                    f"softening_scale_{model_name}": softening_scale,
                    "raw_force_predictions": force_prediction,
                }
            )
            if checkpoint and (len(results) % checkpoint_freq == 0 or material_id == self.material_ids[-1]):
                checkpoint.save(results)

        if include_full_results:
            results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame(
                [{k: v for k, v in res.items() if k != "raw_force_predictions"} for res in results]
            )
        return results_df


class BenchmarkSuite:
    """
    Represents a suite for handling and executing a list of benchmarks. This class is designed
    for the comprehensive execution and management of benchmarks with support for configurable
    parallel computation and checkpointing.

    The purpose of this class is to facilitate the execution of multiple benchmarks using
    various computational models (calculators) while enabling efficient resource utilization
    and result aggregation. It supports checkpointing to handle long computations reliably.

    Attributes:
        benchmarks: Sequence of ``Benchmark`` (or compatible) instances.
    """

    def __init__(self, benchmarks: list) -> None:
        """
        Args:
            benchmarks: Benchmark objects to run in sequence.
        """
        self.benchmarks = benchmarks

    def run(
        self,
        calculators: dict[str, Calculator],
        *,
        n_jobs: int | None = -1,
        checkpoint_freq: int = 1000,
        delete_checkpoint_on_finish: bool = True,
    ) -> list[pd.DataFrame]:
        """
        Executes benchmarks using the provided calculators and combines the results into a
        list of dataframes. Each benchmark runs for all models provided by calculators, collecting
        individual results and joining columns without duplicate reference fields.

        Args:
            calculators: Map of model label to ASE calculator.
            n_jobs: Parallelism forwarded to each ``benchmark.run``.
            checkpoint_freq: Checkpoint interval per benchmark run.
            delete_checkpoint_on_finish: Remove per-model checkpoint files when done.

        Returns:
            One combined DataFrame per benchmark (joined across models).
        """
        all_results = []
        for benchmark in self.benchmarks:
            results: list[pd.DataFrame] = []
            for model_name, calculator in calculators.items():
                chkpt_file = f"{benchmark.__class__.__name__}_{model_name}.csv"
                result = benchmark.run(
                    calculator,
                    model_name,
                    n_jobs=n_jobs,
                    checkpoint_file=chkpt_file,
                    checkpoint_freq=checkpoint_freq,
                    delete_checkpoint_on_finish=delete_checkpoint_on_finish,
                )
                if results:
                    # Remove duplicate DFT columns.
                    todrop = [c for c in result.columns if c in results[0].columns]
                    result = result.drop(todrop, axis=1)
                results.append(result)
            combined = results[0].join(results[1:], validate="one_to_one")
            all_results.append(combined)
        return all_results
