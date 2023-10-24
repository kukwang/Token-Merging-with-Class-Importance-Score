# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch

def do_nothing(x):
    return x

def bipartite_soft_matching_ToMeCIS(
    metric: torch.Tensor,   # [B, N+1, C//H] (include cls token) or [B, N, C//H] (not include cls token)
    r: int,
    class_token: bool = True,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMeCIS with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected = 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # [B, N+1, C//H]
        a, b = metric[..., ::2, :], metric[..., 1::2, :]    # [B, Ne(==(N+1)//2 (even)), C//H], [B, No(==(N+1)//2(odd)), C//H]
        scores = a @ b.transpose(-1, -2)                    # [B, Ne, No]

        if class_token:
            scores[..., 0, :] = -math.inf       # protect cls token, [B, 0, No] = -inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)     # [B, Ne], [B, Ne]
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # [B, Ne, 1]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [B, Ne - r, 1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, [B, r, 1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # [B, r, 1]

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # [B, Ne, C], [B, No, C]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    # [B, Ne-r, C]]
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         # [B, r, C]
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # [B, No, C]

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1) # [B, N+1-r, C]

    return merge

def merge_wavg_score(
    merge: Callable,
    x: torch.Tensor,            # [B, N+1, C]
    score: torch.Tensor,        # [B, N+1, 1]
    size: torch.Tensor = None,  # [B, N+1, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on score.
    Update token size for tome_attention
    Returns the merged tensor and the new token sizes.
    
    Arguments
        x: input tokens, [B, N+1, C]
        score: score used in weighted sum, (e.g., attention value with class token), [B, N+1, 1]
        size: size of tokens (how many tokens are merge), [B, N+1, 1]
    """
    
    if size is None:
        size = torch.ones_like(x[..., 0, None]) # [B, N+1, 1]
    # update token size
    size = merge(size, mode="sum")      # [B, N+1-r, 1]

    # merge tokens
    x = merge(x * score, mode="sum")  # [B, N+1-r, C]
    score = merge(score, mode="sum")  # [B, N+1-r, 1]
    x = x / score   # [B, N+1-r, C]

    return x, size
