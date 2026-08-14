from __future__ import annotations

import torch


def denorm_tensor(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ((x.float() + 1.0) * 0.5 * (high - low) + low).clamp(low, high)


def normalize_for_display(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ((x - low) / (high - low)).clamp(0, 1)


def make_visual_grid(ct: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, norm: dict, max_items: int = 4) -> torch.Tensor:
    from torchvision.utils import make_grid

    ct = ct[:max_items].detach().cpu()
    fake = fake[:max_items].detach().cpu()
    real = real[:max_items].detach().cpu()
    ct_show = normalize_for_display(denorm_tensor(ct, norm["ct_min"], norm["ct_max"]), norm["ct_min"], norm["ct_max"])
    fake_suv = denorm_tensor(fake, norm["pet_min"], norm["pet_max"])
    real_suv = denorm_tensor(real, norm["pet_min"], norm["pet_max"])
    fake_show = normalize_for_display(fake_suv, norm["pet_min"], norm["pet_max"])
    real_show = normalize_for_display(real_suv, norm["pet_min"], norm["pet_max"])
    err_show = normalize_for_display(torch.abs(fake_suv - real_suv), 0.0, norm["pet_max"])
    rows = torch.cat([ct_show, fake_show, real_show, err_show], dim=0)
    return make_grid(rows, nrow=max_items, padding=2)
