from __future__ import annotations

import csv
import json
import logging
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def case_id(path: str | Path) -> str:
    name = Path(path).name
    match = re.search(r"\d+", name)
    if not match:
        raise ValueError(f"Could not extract numeric case id from {name}")
    return match.group(0)


def pair_cases(ct_dir: Path, pet_dir: Path) -> list[dict[str, str]]:
    ct = {case_id(p): p for p in sorted(ct_dir.glob("*.nii*"))}
    pet = {case_id(p): p for p in sorted(pet_dir.glob("*.nii*"))}
    ids = sorted(set(ct) & set(pet), key=lambda x: int(x))
    missing_ct = sorted(set(pet) - set(ct))
    missing_pet = sorted(set(ct) - set(pet))
    if missing_ct or missing_pet:
        logging.warning("Unpaired cases. missing_ct=%s missing_pet=%s", missing_ct, missing_pet)
    return [{"case_id": i, "ct": str(ct[i]), "pet": str(pet[i])} for i in ids]


def normalize_to_minus_one_one(x: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), low, high)
    return ((x - low) / (high - low) * 2.0 - 1.0).astype(np.float16)


def denormalize_from_minus_one_one(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return ((x.astype(np.float32) + 1.0) * 0.5 * (high - low) + low).clip(low, high)


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024**3)


def ensure_cache_budget(path: Path, max_gb: float) -> None:
    size = dir_size_gb(path)
    if size > max_gb:
        raise RuntimeError(f"Staging cache {path} is {size:.2f}GB, above max_cache_gb={max_gb}")


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
