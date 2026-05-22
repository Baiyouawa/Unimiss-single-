from pathlib import Path

import tsdb
from benchpots.datasets import preprocess_ett, preprocess_italy_air_quality

from preprocessing.constants import ETT, IAQ
from common.paths import DATA_ROOT


def ensure_cache_dir() -> Path:
    cache_dir = DATA_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tsdb.data_processing as tsdb_data_processing

        tsdb_data_processing.CACHED_DATASET_DIR = str(cache_dir.resolve())
    except Exception:
        pass
    return cache_dir.resolve()


def ensure_tsdb_cache(dataset_name: str) -> Path:
    cache_dir = ensure_cache_dir()
    dataset_dir = cache_dir / dataset_name
    if not dataset_dir.exists():
        legacy_cache = Path.home() / ".pypots" / "tsdb"
        if legacy_cache.exists():
            try:
                tsdb.migrate_cache(str(cache_dir))
            except FileExistsError:
                pass
    return dataset_dir


def load_dataset_splits(dataset_name: str, prep_n_steps: int) -> dict:
    ensure_tsdb_cache(dataset_name)
    if dataset_name == ETT:
        return preprocess_ett(subset="ETTm2", rate=0.01, n_steps=prep_n_steps, pattern="point")
    if dataset_name == IAQ:
        return preprocess_italy_air_quality(rate=0.01, n_steps=prep_n_steps, pattern="point")
    raise ValueError(f"Unsupported dataset: {dataset_name}")
