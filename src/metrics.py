from __future__ import annotations

import math

import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 50.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse)


def ssim_simple(pred: torch.Tensor, target: torch.Tensor, data_range: float = 50.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        if math.isfinite(value):
            self.total += value * n
            self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
