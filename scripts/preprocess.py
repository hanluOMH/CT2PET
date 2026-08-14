#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import ChunkManager
from src.utils import load_config, pair_cases, setup_logging


def resolve_paths(cfg: dict) -> tuple[Path, Path]:
    data_root = (ROOT / cfg["paths"]["data_root"]).resolve()
    staging_root = (ROOT / cfg["paths"]["staging_root"]).resolve()
    return data_root, staging_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one or more PSMA Pix2Pix staging chunks.")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--chunks", type=int, default=1, help="Number of chunks to prepare from the start of each split.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    data_root, staging_root = resolve_paths(cfg)

    splits = []
    if args.split in ("train", "all"):
        splits.append(
            (
                "train",
                pair_cases(data_root / cfg["paths"]["train_ct_dir"], data_root / cfg["paths"]["train_pet_dir"]),
            )
        )
    if args.split in ("val", "all"):
        splits.append(("val", pair_cases(data_root / cfg["paths"]["val_ct_dir"], data_root / cfg["paths"]["val_pet_dir"])))

    for split, cases in splits:
        manager = ChunkManager(cases, split, staging_root, cfg)
        logging.info("%s has %d cases in %d chunks", split, len(cases), len(manager))
        for idx in range(min(args.chunks, len(manager))):
            path = manager.prepare_sync(idx)
            logging.info("Prepared %s", path)
        manager.close()


if __name__ == "__main__":
    main()
