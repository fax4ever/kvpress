# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class KeyDiffPress(ScorerPress):
    """
    KeyDiff: Key similarity-based KV cache compression.

    Evicts tokens based on key vector similarity to average key pattern.
    Identifies tokens with most similar keys to average and removes them,
    keeping tokens with more distinctive key vectors.

    Based on KeyDiff (https://arxiv.org/abs/2504.15364).

    Note: The original press in the KeyDiff paper implements a block-wise iterative compression.
    In KVPress, the iterative compression is implemented in the BlockPress class.
    Therefore, to replicate the paper's implementation, please use:

    `press = BlockPress(press=KeyDiffPress(compression_ratio=0.x), block_size=N)`

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        valid_mask = kwargs.get("valid_mask") if isinstance(kwargs, dict) else None
        normalized = F.normalize(keys, p=2, dim=-1)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            anchor = normalized.masked_fill(~mask, 0).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True).clamp(min=1)
        else:
            anchor = normalized.mean(dim=2, keepdim=True)

        return -F.cosine_similarity(keys, anchor, dim=-1)
