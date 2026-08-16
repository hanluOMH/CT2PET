#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.cache import ChunkManager
from src.data import SliceDataset, create_loader
from src.metrics import AverageMeter, mae, psnr, rmse, ssim_simple
from src.models import PatchDiscriminator, UNetGenerator, init_weights
from src.utils import append_metrics, load_config, pair_cases, safe_rmtree, set_seed, setup_logging
from src.visualization import denorm_tensor, make_visual_grid, save_visual_grid


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def gan_loss(pred: torch.Tensor, target_is_real: bool, mode: str) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    if mode == "lsgan":
        return nn.functional.mse_loss(pred, target)
    return nn.functional.binary_cross_entropy_with_logits(pred, target)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler: Any,
    best_mae: float,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae": best_mae,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1, int(ckpt.get("global_step", 0)), float(ckpt.get("best_mae", float("inf")))


@torch.no_grad()
def validate(
    generator: nn.Module,
    val_cases: list[dict[str, str]],
    cfg: dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> dict[str, float]:
    generator.eval()
    staging_root = resolve_path(cfg["paths"]["staging_root"])
    manager = ChunkManager(val_cases, "val", staging_root, cfg)
    meters = {k: AverageMeter() for k in ["l1_norm", "mae", "rmse", "psnr", "ssim"]}
    visual_ct: list[torch.Tensor] = []
    visual_fake: list[torch.Tensor] = []
    visual_real: list[torch.Tensor] = []
    visual_seen = 0
    visual_limit = int(cfg["data"]["val_visuals"])

    for chunk_idx in range(len(manager)):
        chunk_dir = manager.prepare_sync(chunk_idx)
        dataset = SliceDataset(
            chunk_dir=chunk_dir,
            image_size=cfg["data"]["image_size"],
            slice_axis=cfg["data"]["slice_axis"],
            skip_empty_slices=cfg["data"]["skip_empty_slices"],
            empty_pet_threshold=cfg["data"]["empty_pet_threshold"],
            max_slices=cfg["data"]["max_val_slices"],
        )
        loader = create_loader(dataset, cfg, shuffle=False)
        pbar = tqdm(loader, desc=f"val epoch {epoch} chunk {chunk_idx + 1}/{len(manager)}", leave=False)
        for batch in pbar:
            ct = batch["ct"].to(device, non_blocking=True)
            real = batch["pet"].to(device, non_blocking=True)
            fake = generator(ct)
            l1 = nn.functional.l1_loss(fake, real)
            fake_suv = denorm_tensor(fake, cfg["normalization"]["pet_min"], cfg["normalization"]["pet_max"])
            real_suv = denorm_tensor(real, cfg["normalization"]["pet_min"], cfg["normalization"]["pet_max"])
            n = ct.size(0)
            meters["l1_norm"].update(float(l1.item()), n)
            meters["mae"].update(float(mae(fake_suv, real_suv).item()), n)
            meters["rmse"].update(float(rmse(fake_suv, real_suv).item()), n)
            meters["psnr"].update(float(psnr(fake_suv, real_suv).item()), n)
            meters["ssim"].update(float(ssim_simple(fake_suv, real_suv).item()), n)
            for item_idx in range(n):
                visual_seen += 1
                if len(visual_ct) < visual_limit:
                    visual_ct.append(ct[item_idx].cpu())
                    visual_fake.append(fake[item_idx].cpu())
                    visual_real.append(real[item_idx].cpu())
                else:
                    replace_idx = random.randint(0, visual_seen - 1)
                    if replace_idx < visual_limit:
                        visual_ct[replace_idx] = ct[item_idx].cpu()
                        visual_fake[replace_idx] = fake[item_idx].cpu()
                        visual_real[replace_idx] = real[item_idx].cpu()
        manager.cleanup(chunk_idx)
    manager.close()

    results = {k: v.avg for k, v in meters.items()}
    for key, value in results.items():
        writer.add_scalar(f"val/{key}", value, epoch)
    if visual_ct:
        grid = make_visual_grid(
            torch.stack(visual_ct),
            torch.stack(visual_fake),
            torch.stack(visual_real),
            cfg["normalization"],
            max_items=int(cfg["data"]["val_visuals"]),
        )
        writer.add_image("val/ct_fake_real_error", grid, epoch)
        writer.add_image(f"val_epoch_{epoch:03d}/ct_fake_real_error", grid, epoch)
        writer.flush()
        image_dir = resolve_path(cfg["paths"].get("validation_image_dir", "validation_images"))
        save_visual_grid(image_path, grid)
        logging.info("Saved validation image grid to %s", image_path)
    generator.train()
    return results


def train_one_epoch(
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler: Any,
    train_cases: list[dict[str, str]],
    cfg: dict[str, Any],
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[int, dict[str, float]]:
    generator.train()
    discriminator.train()
    staging_root = resolve_path(cfg["paths"]["staging_root"])
    manager = ChunkManager(train_cases, "train", staging_root, cfg)
    meters = {k: AverageMeter() for k in ["loss_g", "loss_d", "loss_l1", "loss_gan"]}
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    if len(manager) > 1:
        manager.prefetch(0)
    for chunk_idx in range(len(manager)):
        chunk_dir = manager.get_prefetched_or_prepare(chunk_idx)
        if chunk_idx + 1 < len(manager):
            manager.prefetch(chunk_idx + 1)
        dataset = SliceDataset(
            chunk_dir=chunk_dir,
            image_size=cfg["data"]["image_size"],
            slice_axis=cfg["data"]["slice_axis"],
            skip_empty_slices=cfg["data"]["skip_empty_slices"],
            empty_pet_threshold=cfg["data"]["empty_pet_threshold"],
            max_slices=cfg["data"]["max_train_slices"],
        )
        loader = create_loader(dataset, cfg, shuffle=True)
        pbar = tqdm(loader, desc=f"train epoch {epoch} chunk {chunk_idx + 1}/{len(manager)}")
        for batch_idx, batch in enumerate(pbar, start=1):
            ct = batch["ct"].to(device, non_blocking=True)
            real = batch["pet"].to(device, non_blocking=True)

            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                fake = generator(ct).detach()
                loss_d = 0.5 * (
                    gan_loss(discriminator(ct, real), True, cfg["model"]["gan_mode"])
                    + gan_loss(discriminator(ct, fake), False, cfg["model"]["gan_mode"])
                )
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)

            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                fake = generator(ct)
                loss_gan = gan_loss(discriminator(ct, fake), True, cfg["model"]["gan_mode"])
                loss_l1 = nn.functional.l1_loss(fake, real)
                loss_g = loss_gan + float(cfg["model"]["lambda_l1"]) * loss_l1
            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            n = ct.size(0)
            global_step += 1
            meters["loss_g"].update(float(loss_g.item()), n)
            meters["loss_d"].update(float(loss_d.item()), n)
            meters["loss_l1"].update(float(loss_l1.item()), n)
            meters["loss_gan"].update(float(loss_gan.item()), n)
            if global_step % int(cfg["train"]["log_every"]) == 0:
                writer.add_scalar("train/loss_g", float(loss_g.item()), global_step)
                writer.add_scalar("train/loss_d", float(loss_d.item()), global_step)
                writer.add_scalar("train/loss_l1", float(loss_l1.item()), global_step)
            pbar.set_postfix(
                loss_g=f"{loss_g.item():.4f}",
                loss_d=f"{loss_d.item():.4f}",
                l1=f"{loss_l1.item():.4f}",
            )
        manager.cleanup(chunk_idx)
    manager.close()
    return global_step, {k: v.avg for k, v in meters.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pix2Pix for PSMA CT-to-PET generation.")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    device = get_device(cfg["train"]["device"])
    logging.info("Using device: %s", device)

    data_root = resolve_path(cfg["paths"]["data_root"])
    train_cases = pair_cases(data_root / cfg["paths"]["train_ct_dir"], data_root / cfg["paths"]["train_pet_dir"])
    val_cases = pair_cases(data_root / cfg["paths"]["val_ct_dir"], data_root / cfg["paths"]["val_pet_dir"])
    logging.info("Found %d train cases and %d val cases", len(train_cases), len(val_cases))
    if not train_cases or not val_cases:
        raise RuntimeError("Need at least one train case and one validation case")

    model_cfg = cfg["model"]
    generator = UNetGenerator(model_cfg["in_channels"], model_cfg["out_channels"], model_cfg["base_channels"]).to(device)
    discriminator = PatchDiscriminator(model_cfg["in_channels"], model_cfg["out_channels"], model_cfg["base_channels"]).to(device)
    init_weights(generator)
    init_weights(discriminator)

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg["train"]["lr"]), betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["train"]["lr"]), betas=(float(cfg["train"]["beta1"]), float(cfg["train"]["beta2"])))
    scaler = make_grad_scaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    checkpoint_dir = resolve_path(cfg["paths"]["checkpoint_dir"])
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["project"]["name"]
    metrics_file = resolve_path(cfg["paths"]["metrics_file"])
    writer = SummaryWriter(log_dir=str(run_dir))
    start_epoch = 1
    global_step = 0
    best_mae = float("inf")
    latest = checkpoint_dir / "latest.pt"
    if bool(cfg["train"]["resume"]) and latest.exists():
        start_epoch, global_step, best_mae = load_checkpoint(latest, generator, discriminator, opt_g, opt_d, scaler, device)
        logging.info("Resumed from %s. Starting epoch %d", latest, start_epoch)
    else:
        safe_rmtree(resolve_path(cfg["paths"]["staging_root"]))

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        logging.info("Starting epoch %d/%d", epoch, int(cfg["train"]["epochs"]))
        global_step, train_metrics = train_one_epoch(
            epoch, generator, discriminator, opt_g, opt_d, scaler, train_cases, cfg, device, writer, global_step
        )
        val_metrics = validate(generator, val_cases, cfg, device, writer, epoch)
        for key, value in train_metrics.items():
            writer.add_scalar(f"epoch_train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"epoch_val/{key}", value, epoch)

        logging.info(
            "Epoch %d summary | train G %.4f D %.4f L1 %.4f | val MAE %.4f RMSE %.4f PSNR %.4f SSIM %.4f",
            epoch,
            train_metrics["loss_g"],
            train_metrics["loss_d"],
            train_metrics["loss_l1"],
            val_metrics["mae"],
            val_metrics["rmse"],
            val_metrics["psnr"],
            val_metrics["ssim"],
        )
        append_metrics(metrics_file, {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

        save_checkpoint(latest, epoch, global_step, generator, discriminator, opt_g, opt_d, scaler, best_mae, cfg)
        if bool(cfg["train"]["save_epoch_checkpoints"]):
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", epoch, global_step, generator, discriminator, opt_g, opt_d, scaler, best_mae, cfg)
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            save_checkpoint(checkpoint_dir / "best.pt", epoch, global_step, generator, discriminator, opt_g, opt_d, scaler, best_mae, cfg)
            logging.info("New best checkpoint: val MAE %.4f", best_mae)

    writer.close()
    logging.info("Training complete")


if __name__ == "__main__":
    main()
