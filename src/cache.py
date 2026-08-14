from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .nifti import read_nifti
from .utils import ensure_cache_budget, normalize_to_minus_one_one, safe_rmtree, write_json


class ChunkManager:
    def __init__(
        self,
        cases: list[dict[str, str]],
        split: str,
        staging_root: Path,
        cfg: dict[str, Any],
    ) -> None:
        self.cases = cases
        self.split = split
        self.staging_root = staging_root
        self.chunk_cases = int(cfg["data"]["chunk_cases"])
        self.max_cache_gb = float(cfg["data"]["max_cache_gb"])
        self.cfg = cfg
        self.norm = cfg["normalization"]
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future[Path] | None = None
        self._future_index: int | None = None

    def __len__(self) -> int:
        if not self.cases:
            return 0
        return (len(self.cases) + self.chunk_cases - 1) // self.chunk_cases

    def chunk_cases_for(self, index: int) -> list[dict[str, str]]:
        start = index * self.chunk_cases
        return self.cases[start : start + self.chunk_cases]

    def chunk_dir(self, index: int) -> Path:
        return self.staging_root / self.split / f"chunk_{index:04d}"

    def prepare_sync(self, index: int) -> Path:
        return self._prepare(index)

    def prefetch(self, index: int) -> None:
        if index >= len(self):
            return
        if self._future is not None and self._future_index == index:
            return
        self._future = self._executor.submit(self._prepare, index)
        self._future_index = index

    def get_prefetched_or_prepare(self, index: int) -> Path:
        if self._future is not None and self._future_index == index:
            return self._future.result()
        return self.prepare_sync(index)

    def cleanup(self, index: int) -> None:
        safe_rmtree(self.chunk_dir(index))

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _prepare(self, index: int) -> Path:
        out_dir = self.chunk_dir(index)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            return out_dir

        safe_rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_manifest: list[dict[str, Any]] = []
        logging.info("Preparing %s chunk %d with %d cases", self.split, index, len(self.chunk_cases_for(index)))
        for case in self.chunk_cases_for(index):
            ct, ct_meta = read_nifti(case["ct"])
            pet, pet_meta = read_nifti(case["pet"])
            if ct.shape != pet.shape:
                raise ValueError(f"Shape mismatch for case {case['case_id']}: CT {ct.shape} PET {pet.shape}")

            ct_norm = normalize_to_minus_one_one(ct, self.norm["ct_min"], self.norm["ct_max"])
            pet_norm = normalize_to_minus_one_one(pet, self.norm["pet_min"], self.norm["pet_max"])
            original_shape = tuple(ct_norm.shape)
            slice_axis = int(self.cfg["data"].get("slice_axis", 2)) if hasattr(self, "cfg") else 2
            if slice_axis != 0:
                ct_norm = np.moveaxis(ct_norm, slice_axis, 0)
                pet_norm = np.moveaxis(pet_norm, slice_axis, 0)
            ct_norm = np.ascontiguousarray(ct_norm)
            pet_norm = np.ascontiguousarray(pet_norm)
            ct_path = out_dir / f"{case['case_id']}_ct.npy"
            pet_path = out_dir / f"{case['case_id']}_pet.npy"
            np.save(ct_path, ct_norm)
            np.save(pet_path, pet_norm)
            chunk_manifest.append(
                {
                    "case_id": case["case_id"],
                    "ct": str(ct_path),
                    "pet": str(pet_path),
                    "shape": list(ct_norm.shape),
                    "original_shape": list(original_shape),
                    "cached_slice_axis": 0,
                    "ct_meta": ct_meta,
                    "pet_meta": pet_meta,
                }
            )
            del ct, pet, ct_norm, pet_norm

        write_json(manifest_path, chunk_manifest)
        ensure_cache_budget(self.staging_root, self.max_cache_gb)
        return out_dir
