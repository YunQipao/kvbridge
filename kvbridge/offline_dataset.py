"""Read prepared Hugging Face Arrow caches without contacting the Hub."""
import json
import os
from pathlib import Path


def cached_dataset_splits(dataset, split, config=None):
    from datasets import Dataset, concatenate_datasets
    from datasets import config as datasets_config
    from datasets.naming import camelcase_to_snakecase

    # Use the same cache root and dataset directory naming as datasets itself.
    root = Path(os.environ.get("HF_DATASETS_CACHE") or
                (str(Path(os.environ["HF_HOME"]) / "datasets") if os.environ.get("HF_HOME")
                 else str(datasets_config.HF_DATASETS_CACHE))).expanduser()
    parts = dataset.split("/")
    parts[-1] = camelcase_to_snakecase(parts[-1])
    folder = root / "___".join(parts)
    if config:
        names = [config]
    else:
        names = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    if not names:
        raise FileNotFoundError(f"No prepared dataset cache at {folder}; download the dataset before offline evaluation")
    for name in names:
        candidates = []
        for info_path in (folder / name).glob("*/*/dataset_info.json"):
            info = json.loads(info_path.read_text())
            if info.get("config_name") != name or split not in info.get("splits", {}):
                continue
            prefix = camelcase_to_snakecase(info.get("dataset_name", parts[-1]))
            directory = info_path.parent
            files = sorted(directory.glob(f"{prefix}-{split}-*.arrow"))
            single = directory / f"{prefix}-{split}.arrow"
            if single.is_file():
                files = [single]
            if files:
                candidates.append((info_path.stat().st_mtime_ns, str(directory), info, files))
        if not candidates:
            raise FileNotFoundError(f"Missing prepared split {split!r} for config {name!r} in {folder}; no network fallback attempted")
        _, _, info, files = max(candidates, key=lambda item: item[:2])
        tables = [Dataset.from_file(str(path)) for path in files]
        ds = tables[0] if len(tables) == 1 else concatenate_datasets(tables)
        expected = info["splits"][split].get("num_examples")
        if expected is not None and len(ds) != expected:
            raise RuntimeError(f"Incomplete cached split {name}/{split}: {len(ds)} rows, expected {expected}")
        yield name, ds
