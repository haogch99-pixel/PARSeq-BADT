from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from strhub.models.base import CrossEntropySystem
from strhub.models.parseq.model import PARSeq as Model

from .augment import BADTDegradation


class PARSeq(CrossEntropySystem):
    """PARSeq trained with Boundary-Aware Degradation Training (BADT)."""

    def __init__(
        self,
        charset_train: str,
        charset_test: str,
        max_label_length: int,
        batch_size: int,
        lr: float,
        warmup_pct: float,
        weight_decay: float,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        perm_num: int,
        perm_forward: bool,
        perm_mirrored: bool,
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
        badt_degradation_prob: float = 0.60,
        badt_max_ops: int = 2,
        badt_consistency_weight: float = 0.15,
        badt_confidence_threshold: float = 0.75,
        badt_temperature: float = 1.0,
        badt_require_teacher_correct: bool = True,
        badt_lowres_min_scale: float = 0.35,
        badt_lowres_max_scale: float = 0.75,
        badt_boundary_max_ratio: float = 0.04,
        badt_boundary_token_ratio: float = 0.10,
        badt_boundary_token_drop_prob: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay
        )
        self.save_hyperparameters()

        if badt_consistency_weight < 0:
            raise ValueError("badt_consistency_weight must be >= 0")
        if not 0 <= badt_confidence_threshold <= 1:
            raise ValueError("badt_confidence_threshold must be in [0, 1]")
        if badt_temperature <= 0:
            raise ValueError("badt_temperature must be > 0")
        if not 0 <= badt_boundary_token_ratio <= 0.5:
            raise ValueError("badt_boundary_token_ratio must be in [0, 0.5]")
        if not 0 <= badt_boundary_token_drop_prob <= 1:
            raise ValueError("badt_boundary_token_drop_prob must be in [0, 1]")

        self.model = Model(
            len(self.tokenizer),
            max_label_length,
            img_size,
            patch_size,
            embed_dim,
            enc_num_heads,
            enc_mlp_ratio,
            enc_depth,
            dec_num_heads,
            dec_mlp_ratio,
            dec_depth,
            decode_ar,
            refine_iters,
            dropout,
        )

        # Make PARSeq permutation sampling follow the global training seed.
        self.rng = np.random.default_rng(
            int(torch.initial_seed() % (2**32))
        )
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        self.badt_augment = BADTDegradation(
            probability=badt_degradation_prob,
            max_ops=badt_max_ops,
            lowres_min_scale=badt_lowres_min_scale,
            lowres_max_scale=badt_lowres_max_scale,
            boundary_max_ratio=badt_boundary_max_ratio,
        )
        self.badt_consistency_weight = badt_consistency_weight
        self.badt_confidence_threshold = badt_confidence_threshold
        self.badt_temperature = badt_temperature
        self.badt_require_teacher_correct = badt_require_teacher_correct

        self.patch_rows = int(img_size[0]) // int(patch_size[0])
        self.patch_cols = int(img_size[1]) // int(patch_size[1])
        self.badt_boundary_token_ratio = badt_boundary_token_ratio
        self.badt_boundary_token_drop_prob = badt_boundary_token_drop_prob

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        return self.model.forward(self.tokenizer, images, max_length)

    def gen_tgt_perms(self, tgt: Tensor) -> Tensor:
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)

        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                indices = self.rng.choice(
                    len(perm_pool), size=num_gen_perms - len(perms), replace=False
                )
                perms = torch.cat([perms, perm_pool[indices]])
        else:
            perms.extend(
                [
                    torch.randperm(max_num_chars, device=self._device)
                    for _ in range(num_gen_perms - len(perms))
                ]
            )
            perms = torch.stack(perms)

        if self.perm_mirrored:
            complementary = perms.flip(-1)
            perms = (
                torch.stack([perms, complementary])
                .transpose(0, 1)
                .reshape(-1, max_num_chars)
            )

        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(
                max_num_chars + 1, device=self._device
            )
        return perms

    def generate_attn_masks(self, perm: Tensor) -> tuple[Tensor, Tensor]:
        size = perm.shape[0]
        mask = torch.zeros((size, size), dtype=torch.bool, device=self._device)
        for index in range(size):
            query_idx = perm[index]
            masked_keys = perm[index + 1 :]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(size, dtype=torch.bool, device=self._device)] = True
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def _boundary_token_dropout(self, memory: Tensor) -> Tensor:
        if (
            not self.training
            or self.badt_boundary_token_drop_prob <= 0
            or self.badt_boundary_token_ratio <= 0
        ):
            return memory

        batch_size, num_tokens, embed_dim = memory.shape
        expected_tokens = self.patch_rows * self.patch_cols
        if num_tokens != expected_tokens:
            return memory

        boundary_cols = max(
            1, int(math.ceil(self.patch_cols * self.badt_boundary_token_ratio))
        )
        boundary = torch.zeros(
            batch_size,
            self.patch_rows,
            self.patch_cols,
            1,
            dtype=torch.bool,
            device=memory.device,
        )
        boundary[:, :, :boundary_cols] = True
        boundary[:, :, -boundary_cols:] = True

        drop = (
            torch.rand(
                batch_size,
                self.patch_rows,
                self.patch_cols,
                1,
                device=memory.device,
            )
            < self.badt_boundary_token_drop_prob
        ) & boundary

        reshaped = memory.reshape(
            batch_size, self.patch_rows, self.patch_cols, embed_dim
        )
        return reshaped.masked_fill(drop, 0).reshape(
            batch_size, num_tokens, embed_dim
        )

    def _compute_badt_loss(
        self, images: Tensor, labels: list[str]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        tgt = self.tokenizer.encode(labels, self._device)
        degraded_images = self.badt_augment(images)

        clean_memory = self.model.encode(images)
        degraded_memory = self.model.encode(degraded_images)
        degraded_memory = self._boundary_token_dropout(degraded_memory)

        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        recognition_sum = images.new_zeros(())
        recognition_numel = 0
        consistency_sum = images.new_zeros((), dtype=torch.float32)
        consistency_numel = images.new_zeros((), dtype=torch.float32)
        valid_numel = images.new_zeros((), dtype=torch.float32)

        temperature = self.badt_temperature

        for perm_index, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)

            clean_out = self.model.decode(
                tgt_in,
                clean_memory,
                tgt_mask,
                tgt_padding_mask,
                tgt_query_mask=query_mask,
            )
            degraded_out = self.model.decode(
                tgt_in,
                degraded_memory,
                tgt_mask,
                tgt_padding_mask,
                tgt_query_mask=query_mask,
            )

            clean_logits = self.model.head(clean_out)
            degraded_logits = self.model.head(degraded_out)

            clean_ce = F.cross_entropy(
                clean_logits.flatten(end_dim=1),
                tgt_out.flatten(),
                ignore_index=self.pad_id,
                reduction="sum",
            )
            degraded_ce = F.cross_entropy(
                degraded_logits.flatten(end_dim=1),
                tgt_out.flatten(),
                ignore_index=self.pad_id,
                reduction="sum",
            )

            num_valid = int((tgt_out != self.pad_id).sum().item())
            recognition_sum = recognition_sum + 0.5 * (clean_ce + degraded_ce)
            recognition_numel += num_valid

            valid_mask = tgt_out != self.pad_id
            teacher_probs = F.softmax(
                clean_logits.detach().float() / temperature, dim=-1
            )
            teacher_confidence, teacher_prediction = teacher_probs.max(dim=-1)
            confidence_mask = teacher_confidence >= self.badt_confidence_threshold
            if self.badt_require_teacher_correct:
                confidence_mask = confidence_mask & (teacher_prediction == tgt_out)
            consistency_mask = valid_mask & confidence_mask

            student_log_probs = F.log_softmax(
                degraded_logits.float() / temperature, dim=-1
            )
            token_kl = F.kl_div(
                student_log_probs, teacher_probs, reduction="none"
            ).sum(dim=-1)
            token_kl = token_kl * (temperature**2)

            consistency_sum = consistency_sum + (
                token_kl * consistency_mask.float()
            ).sum()
            consistency_numel = consistency_numel + consistency_mask.sum().float()
            valid_numel = valid_numel + valid_mask.sum().float()

            if perm_index == 1:
                tgt_out = torch.where(
                    tgt_out == self.eos_id, self.pad_id, tgt_out
                )

        recognition_loss = recognition_sum / max(recognition_numel, 1)
        consistency_loss = consistency_sum / consistency_numel.clamp_min(1.0)
        total_loss = (
            recognition_loss
            + self.badt_consistency_weight * consistency_loss
        )

        stats = {
            "recognition_loss": recognition_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "confidence_ratio": (
                consistency_numel / valid_numel.clamp_min(1.0)
            ).detach(),
            "augmentation_l1": (
                degraded_images.detach() - images.detach()
            ).abs().mean(),
        }
        return total_loss, stats

    def training_step(self, batch: tuple[Tensor, list[str]], batch_idx: int) -> STEP_OUTPUT:
        images, labels = batch
        loss, stats = self._compute_badt_loss(images, labels)

        self.log("loss", loss)
        self.log("loss_rec", stats["recognition_loss"])
        self.log("loss_cons", stats["consistency_loss"])
        self.log("badt_conf_ratio", stats["confidence_ratio"])
        self.log("badt_aug_l1", stats["augmentation_l1"])
        return loss
