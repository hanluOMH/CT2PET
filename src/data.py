from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SliceDataset:
    def __init__(
        self,
        chunk_dir: str | Path,
        image_size: int,
        slice_axis: int,
        skip_empty_slices: bool,
        empty_pet_threshold: float,
        max_slices: int | None = None,
    ) -> None:
        try:
            import torch
            from torch.nn import functional as F
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for SliceDataset") from exc

        class _Dataset(Dataset):
            pass

        self._torch = torch
        self._F = F
        self._base_cls = _Dataset
        self.chunk_dir = Path(chunk_dir)
        self.image_size = int(image_size)
        self.slice_axis = int(slice_axis)
        self.volumes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []

        with open(self.chunk_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for volume_idx, item in enumerate(manifest):
            ct = np.load(item["ct"], mmap_mode="r")
            pet = np.load(item["pet"], mmap_mode="r")
            self.volumes.append({"case_id": item["case_id"], "ct": ct, "pet": pet})
            for slice_idx in range(ct.shape[0]):
                if skip_empty_slices:
                    pet_slice = self._take_slice(pet, slice_idx)
                    if float(np.mean(pet_slice > -0.999)) < empty_pet_threshold:
                        continue
                self.index.append((volume_idx, slice_idx))
                if max_slices is not None and len(self.index) >= int(max_slices):
                    return

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        volume_idx, slice_idx = self.index[idx]
        volume = self.volumes[volume_idx]
        ct = self._take_slice(volume["ct"], slice_idx).astype(np.float32)
        pet = self._take_slice(volume["pet"], slice_idx).astype(np.float32)
        ct_t = self._torch.from_numpy(ct.copy())[None, None]
        pet_t = self._torch.from_numpy(pet.copy())[None, None]
        if ct_t.shape[-1] != self.image_size or ct_t.shape[-2] != self.image_size:
            ct_t = self._F.interpolate(ct_t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            pet_t = self._F.interpolate(pet_t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return {
            "ct": ct_t.squeeze(0),
            "pet": pet_t.squeeze(0),
            "case_id": volume["case_id"],
            "slice_idx": slice_idx,
        }

    def _take_slice(self, arr: np.ndarray, idx: int) -> np.ndarray:
        return arr[idx]


def create_loader(dataset, cfg: dict[str, Any], shuffle: bool):
    import torch

    train_cfg = cfg["train"]
    kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": int(train_cfg["num_workers"]),
        "pin_memory": bool(train_cfg["pin_memory"]),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(train_cfg["persistent_workers"])
        if train_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])
    return torch.utils.data.DataLoader(dataset, **kwargs)
