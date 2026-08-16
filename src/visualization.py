from __future__ import annotations

from pathlib import Path

import torch


def denorm_tensor(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ((x.float() + 1.0) * 0.5 * (high - low) + low).clamp(low, high)


def normalize_for_display(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return ((x - low) / (high - low)).clamp(0, 1)


def make_visual_grid(ct: torch.Tensor, fake: torch.Tensor, real: torch.Tensor, norm: dict, max_items: int = 4) -> torch.Tensor:
    """Create one validation slice per row.

    Layout: CT input | Generated PET | Ground-truth PET | Absolute error.
    PET display uses a narrower SUV window than metrics so low uptake remains visible.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    ct = ct[:max_items].detach().cpu()
    fake = fake[:max_items].detach().cpu()
    real = real[:max_items].detach().cpu()
    ct_show = normalize_for_display(denorm_tensor(ct, norm["ct_min"], norm["ct_max"]), norm["ct_min"], norm["ct_max"])
    fake_suv = denorm_tensor(fake, norm["pet_min"], norm["pet_max"])
    real_suv = denorm_tensor(real, norm["pet_min"], norm["pet_max"])
    pet_display_min = float(norm.get("pet_display_min", norm["pet_min"]))
    pet_display_max = float(norm.get("pet_display_max", min(norm["pet_max"], 10.0)))
    error_display_max = float(norm.get("error_display_max", min(norm["pet_max"], 5.0)))
    fake_show = normalize_for_display(fake_suv, pet_display_min, pet_display_max)
    real_show = normalize_for_display(real_suv, pet_display_min, pet_display_max)
    err_show = normalize_for_display(torch.abs(fake_suv - real_suv), 0.0, error_display_max)

    panels: list[torch.Tensor] = []
    for idx in range(ct_show.shape[0]):
        panels.extend([ct_show[idx], fake_show[idx], real_show[idx], err_show[idx]])
    panel_tensor = torch.stack(panels, dim=0)

    labels = ["CT input", "Generated PET (SUV 0-10)", "Ground truth PET (SUV 0-10)", "Absolute error (0-5)"]
    rows = ct_show.shape[0]
    _, height, width = panel_tensor.shape[1:]
    header_h = 34
    label_w = 72
    canvas = Image.new("RGB", (label_w + 4 * width, header_h + rows * height), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        row_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    for col, label in enumerate(labels):
        draw.text((label_w + col * width + 8, 8), label, fill=(255, 255, 255), font=font)
    for row in range(rows):
        draw.text((8, header_h + row * height + 8), f"Slice {row + 1}", fill=(255, 255, 255), font=row_font)
        for col in range(4):
            img_t = panel_tensor[row * 4 + col].squeeze(0).clamp(0, 1)
            arr = (img_t.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            canvas.paste(img, (label_w + col * width, header_h + row * height))

    arr = np.asarray(canvas).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_visual_grid(path: str | Path, grid: torch.Tensor) -> None:
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(path))
