from __future__ import annotations

import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        self.d1 = DownBlock(in_channels, base, normalize=False)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.d4 = DownBlock(base * 4, base * 8)
        self.d5 = DownBlock(base * 8, base * 8)
        self.d6 = DownBlock(base * 8, base * 8)
        self.bottleneck = nn.Sequential(nn.Conv2d(base * 8, base * 8, 4, 2, 1), nn.ReLU(inplace=True))
        self.u1 = UpBlock(base * 8, base * 8, dropout=True)
        self.u2 = UpBlock(base * 16, base * 8, dropout=True)
        self.u3 = UpBlock(base * 16, base * 8)
        self.u4 = UpBlock(base * 16, base * 4)
        self.u5 = UpBlock(base * 8, base * 2)
        self.u6 = UpBlock(base * 4, base)
        self.final = nn.Sequential(nn.ConvTranspose2d(base * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        b = self.bottleneck(d6)
        u1 = self.u1(b, d6)
        u2 = self.u2(u1, d5)
        u3 = self.u3(u2, d4)
        u4 = self.u4(u3, d3)
        u5 = self.u5(u4, d2)
        u6 = self.u6(u5, d1)
        return self.final(u6)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        ch = in_channels + out_channels
        self.net = nn.Sequential(
            nn.Conv2d(ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ct, pet], dim=1))


def init_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)
