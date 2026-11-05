from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BADTDegradation(nn.Module):
    """Online IC15-style degradation for normalized image tensors."""

    NUM_OPS = 5

    def __init__(
        self,
        probability: float = 0.60,
        max_ops: int = 2,
        lowres_min_scale: float = 0.35,
        lowres_max_scale: float = 0.75,
        boundary_max_ratio: float = 0.04,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if max_ops < 1:
            raise ValueError("max_ops must be >= 1")
        if not 0.0 < lowres_min_scale <= lowres_max_scale <= 1.0:
            raise ValueError("invalid low-resolution scale range")
        if not 0.0 <= boundary_max_ratio <= 0.25:
            raise ValueError("boundary_max_ratio must be in [0, 0.25]")

        self.probability = probability
        self.max_ops = max_ops
        self.lowres_min_scale = lowres_min_scale
        self.lowres_max_scale = lowres_max_scale
        self.boundary_max_ratio = boundary_max_ratio

    @staticmethod
    def _merge(original: Tensor, transformed: Tensor, mask: Tensor) -> Tensor:
        return torch.where(mask[:, None, None, None], transformed, original)

    def _low_resolution(self, images: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return images
        height, width = images.shape[-2:]
        scale = torch.empty((), device=images.device).uniform_(
            self.lowres_min_scale, self.lowres_max_scale
        )
        small_h = max(4, int(round(height * float(scale))))
        small_w = max(8, int(round(width * float(scale))))
        low = F.interpolate(
            images, size=(small_h, small_w), mode="bilinear", align_corners=False
        )
        restored = F.interpolate(
            low, size=(height, width), mode="bilinear", align_corners=False
        )
        return self._merge(images, restored, mask)

    def _gaussian_blur(self, images: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return images
        kernel_size = 5
        sigma = torch.empty((), device=images.device).uniform_(0.6, 1.6)
        coords = torch.arange(
            kernel_size, device=images.device, dtype=images.dtype
        ) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma.to(images.dtype) ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        weight = kernel_2d.expand(images.shape[1], 1, -1, -1)
        pad = kernel_size // 2
        blurred = F.conv2d(
            F.pad(images, (pad, pad, pad, pad), mode="reflect"),
            weight,
            groups=images.shape[1],
        )
        return self._merge(images, blurred, mask)

    def _random_affine(self, images: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return images

        batch_size = images.shape[0]
        dtype = images.dtype
        device = images.device

        angle = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(7.0)
        shear = (torch.rand(batch_size, device=device) * 2 - 1) * 0.10
        scale = 0.92 + torch.rand(batch_size, device=device) * 0.16
        tx = (torch.rand(batch_size, device=device) * 2 - 1) * 0.04
        ty = (torch.rand(batch_size, device=device) * 2 - 1) * 0.06

        cos_a = torch.cos(angle) * scale
        sin_a = torch.sin(angle)

        theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos_a.to(dtype)
        theta[:, 0, 1] = (-sin_a + shear).to(dtype)
        theta[:, 1, 0] = sin_a.to(dtype)
        theta[:, 1, 1] = cos_a.to(dtype)
        theta[:, 0, 2] = tx.to(dtype)
        theta[:, 1, 2] = ty.to(dtype)

        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
        )
        theta = torch.where(mask[:, None, None], theta, identity[None])

        grid = F.affine_grid(theta, images.shape, align_corners=False)
        warped = F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return self._merge(images, warped, mask)

    def _photometric_noise(self, images: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return images

        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype

        contrast = (
            0.75 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.50
        ).to(dtype)
        brightness = (
            (torch.rand(batch_size, 1, 1, 1, device=device) * 2 - 1) * 0.15
        ).to(dtype)
        noise_std = (
            0.01 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.07
        ).to(dtype)

        mean = images.mean(dim=(-2, -1), keepdim=True)
        transformed = (images - mean) * contrast + mean + brightness
        transformed = transformed + torch.randn_like(images) * noise_std
        return self._merge(images, transformed, mask)

    def _boundary_clutter(self, images: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()) or self.boundary_max_ratio <= 0:
            return images

        output = images.clone()
        width = images.shape[-1]
        max_strip = max(1, int(round(width * self.boundary_max_ratio)))

        for index in mask.nonzero(as_tuple=False).flatten().tolist():
            strip_width = int(
                torch.randint(1, max_strip + 1, (1,), device=images.device).item()
            )
            side_left = bool(torch.randint(0, 2, (1,), device=images.device).item())

            low = strip_width
            high = max(low + 1, width - 2 * strip_width + 1)
            source_start = int(
                torch.randint(low, high, (1,), device=images.device).item()
            )
            source_end = min(width, source_start + strip_width)
            patch = output[index : index + 1, :, :, source_start:source_end]

            if patch.shape[-1] != strip_width:
                patch = F.interpolate(
                    patch,
                    size=(patch.shape[-2], strip_width),
                    mode="bilinear",
                    align_corners=False,
                )
            if bool(torch.randint(0, 2, (1,), device=images.device).item()):
                patch = patch.flip(-1)

            patch = patch + torch.randn_like(patch) * 0.05
            alpha = float(
                torch.empty((), device=images.device).uniform_(0.25, 0.55)
            )

            if side_left:
                target = output[index : index + 1, :, :, :strip_width]
                output[index : index + 1, :, :, :strip_width] = (
                    (1 - alpha) * target + alpha * patch
                )
            else:
                target = output[index : index + 1, :, :, -strip_width:]
                output[index : index + 1, :, :, -strip_width:] = (
                    (1 - alpha) * target + alpha * patch
                )

        return output

    def forward(self, images: Tensor) -> Tensor:
        if not self.training or self.probability <= 0:
            return images

        batch_size = images.shape[0]
        device = images.device

        active = torch.rand(batch_size, device=device) < self.probability
        if not bool(active.any()):
            return images

        num_ops = torch.randint(
            1, self.max_ops + 1, (batch_size,), device=device
        )
        num_available_ops = (
            self.NUM_OPS if self.boundary_max_ratio > 0 else self.NUM_OPS - 1
        )
        op_ids = torch.randint(
            0, num_available_ops, (batch_size, self.max_ops), device=device
        )
        slot_valid = (
            torch.arange(self.max_ops, device=device)[None, :] < num_ops[:, None]
        )

        output = images
        op_masks = [
            active & (((op_ids == op_id) & slot_valid).any(dim=1))
            for op_id in range(self.NUM_OPS)
        ]

        output = self._low_resolution(output, op_masks[0])
        output = self._gaussian_blur(output, op_masks[1])
        output = self._random_affine(output, op_masks[2])
        output = self._photometric_noise(output, op_masks[3])
        output = self._boundary_clutter(output, op_masks[4])

        return output.clamp(-1.0, 1.0)
