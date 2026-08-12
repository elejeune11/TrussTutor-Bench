import os
import importlib.util
from typing import Dict
from pathlib import Path

# trusstutor_support_code imports
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.schemas import (
    ALLOWED_TOPICS,
    ALLOWED_SOURCE_TYPES,
    ALLOWED_SOURCE_NAMES
)


class TaskLoader:
    # ... (other methods remain the same)

    @staticmethod
    def _validate_metadata(metadata: dict, filepath: Path) -> None:
        """
        Strictly enforces the metadata schema. Raises a ValueError if 
        a template contains unauthorized vocabularies.
        """
        filename = filepath.name

        # 1. Validate Benchmark ID
        if "benchmark_id" not in metadata:
            raise ValueError(f"[{filename}] Missing required 'benchmark_id'.")

        # 2. Validate Topics
        topics = metadata.get("topic", [])
        if not isinstance(topics, list) or len(topics) == 0:
            raise ValueError(f"[{filename}] 'topic' must be a non-empty list.")
            
        for topic in topics:
            if topic not in ALLOWED_TOPICS:
                raise ValueError(f"[{filename}] Unauthorized topic '{topic}'. Allowed: {ALLOWED_TOPICS}")

        # 3. Validate Source Type
        source_type = metadata.get("source_type")
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"[{filename}] Unauthorized source_type '{source_type}'. Allowed: {ALLOWED_SOURCE_TYPES}")

        # 4. Validate Source Name
        source_name = metadata.get("source_name")
        if source_name not in ALLOWED_SOURCE_NAMES:
            raise ValueError(f"[{filename}] Unauthorized source_name '{source_name}'. Allowed: {ALLOWED_SOURCE_NAMES}")

    @classmethod
    def load_directory(cls, directory_path: str) -> Dict[str, BenchmarkTask]:
        """
        Crawls the target directory, dynamically imports valid templates, 
        and returns a registry mapping benchmark_ids to BenchmarkTask objects.
        """
        target_dir = Path(directory_path)
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        task_registry: Dict[str, BenchmarkTask] = {}
        valid_count = 0
        
        print(f"Loading templates from: {target_dir.resolve()}...")

        for filepath in target_dir.glob("*.py"):
            # Skip hidden files and python cache/init files
            if filepath.name.startswith("__") or filepath.name.startswith("."):
                continue

            try:
                # 1. Dynamically load the module
                spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 2. Ensure it is actually a template (must have METADATA)
                if not hasattr(module, 'METADATA'):
                    print(f"Skipping {filepath.name}: Missing METADATA block.")
                    continue

                # 3. Lint the metadata against schemas.py
                cls._validate_metadata(module.METADATA, filepath)

                # 4. Instantiate the immutable payload
                task_payload = BenchmarkTask.from_module(module)
                
                # 5. Register it
                if task_payload.benchmark_id in task_registry:
                    raise ValueError(f"Duplicate benchmark_id found: {task_payload.benchmark_id}")
                    
                task_registry[task_payload.benchmark_id] = task_payload
                valid_count += 1

            except Exception as e:
                # Fail-fast design: Crash the loader so broken templates are never silently skipped
                raise RuntimeError(f"Failed to load template {filepath.name}:\n{str(e)}") from e

        print(f"Successfully loaded {valid_count} tasks into the registry.")
        return task_registry


