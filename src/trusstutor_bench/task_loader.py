import importlib.util
from typing import Dict
from pathlib import Path

from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.schemas import (
    ALLOWED_TOPICS,
    ALLOWED_SOURCE_TYPES,
    ALLOWED_SOURCE_NAMES
)


class TaskLoader:
    """
    Loads task template files from a directory.
    """

    @staticmethod
    def _validate_metadata(metadata: dict, filepath: Path) -> None:
        """
        Checks a template's metadata against the allowed vocabularies in schemas.py.
        Raises ValueError if the benchmark ID is missing or a value is not allowed.

        Args:
            metadata (dict): The METADATA block of a template.
            filepath (Path): Template file, used in error messages.
        """
        filename = filepath.name

        if "benchmark_id" not in metadata:
            raise ValueError(f"[{filename}] Missing required 'benchmark_id'.")

        topics = metadata.get("topic", [])
        if not isinstance(topics, list) or len(topics) == 0:
            raise ValueError(f"[{filename}] 'topic' must be a non-empty list.")

        for topic in topics:
            if topic not in ALLOWED_TOPICS:
                raise ValueError(f"[{filename}] Unauthorized topic '{topic}'. Allowed: {ALLOWED_TOPICS}")

        source_type = metadata.get("source_type")
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"[{filename}] Unauthorized source_type '{source_type}'. Allowed: {ALLOWED_SOURCE_TYPES}")

        source_name = metadata.get("source_name")
        if source_name not in ALLOWED_SOURCE_NAMES:
            raise ValueError(f"[{filename}] Unauthorized source_name '{source_name}'. Allowed: {ALLOWED_SOURCE_NAMES}")

    @classmethod
    def load_directory(cls, directory_path: str) -> Dict[str, BenchmarkTask]:
        """
        Imports every task template in a directory and validates its metadata.

        Args:
            directory_path (str): Folder containing the template files.

        Returns:
            dict: Benchmark ID -> BenchmarkTask.
        """
        target_dir = Path(directory_path)
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        task_registry: Dict[str, BenchmarkTask] = {}
        valid_count = 0

        print(f"Loading templates from: {target_dir.resolve()}...")

        for filepath in target_dir.glob("*.py"):
            if filepath.name.startswith("__") or filepath.name.startswith("."):
                continue

            try:
                spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, 'METADATA'):
                    print(f"Skipping {filepath.name}: Missing METADATA block.")
                    continue

                cls._validate_metadata(module.METADATA, filepath)

                task_payload = BenchmarkTask.from_module(module)

                if task_payload.benchmark_id in task_registry:
                    raise ValueError(f"Duplicate benchmark_id found: {task_payload.benchmark_id}")

                task_registry[task_payload.benchmark_id] = task_payload
                valid_count += 1

            except Exception as e:
                # Fail fast so broken templates are never silently skipped.
                raise RuntimeError(f"Failed to load template {filepath.name}:\n{str(e)}") from e

        print(f"Successfully loaded {valid_count} tasks into the registry.")
        return task_registry
