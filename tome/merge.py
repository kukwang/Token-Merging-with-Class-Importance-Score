# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # [B, N, C//H]
        a, b = metric[..., ::2, :], metric[..., 1::2, :]    # [B, N//2 (even), C//H], [B, N//2 (odd), C//H]
        scores = a @ b.transpose(-1, -2)                    # [B, N//2 (even), N//2(odd)]

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)     # [B, N//2 (even)], [B, N//2 (even)]
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # [B, N//2 (even), 1]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [B, N//2 (even) - r, 1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, [B, r, 1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # [B, r, 1]

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # [B, N//2 (even), C], [B, N//2 (odd), C]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    # [B, N//2 (even) - r, C]]
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         # [B, r, C]
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) # [B, N//2 (odd), C]

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1) # [B, N - r, C]

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_revised(
    metric: torch.Tensor,   # [B, N+1, C//H] (include cls token) or [B, N, C//H] (not include cls token)
    r: int,
    class_token: bool = True,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

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

    # def merge_idx(x: torch.Tensor, mode="mean") -> torch.Tensor:
    #     src, dst = x[..., ::2, :], x[..., 1::2, :]  # [B, Ne, C], [B, No, C]
    #     n, t1, c = src.shape
    #     unm = src.gather(dim=-2, index=unm_idx.expand(n, t1-r, c))    # [B, Ne-r, C]]

    #     return torch.cat([unm, dst], dim=1) # [B, N-r, C]

    # return merge, merge_idx
    return merge

# get topk indices in sorted order, K == left_tokens, [B, K]
def bipartite_soft_matching_clsattn(
    metric: torch.Tensor,   # [B, N+1, C//H]
    cls_attn: torch.Tensor, # [B, N+1, 1]
    r: int,                 # r
    left_tokens,            # K
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a topk attn score tokens (h_idx) and other tokens (l_idx).
    cls token is not exists in h_idx and l_idx

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    with torch.no_grad():
        # split tokens into higher and lower cls_attn value
        metric_idx = cls_attn.argsort(dim=-2, descending=True)                  # [B, N+1, 1]

        # h_idx[0]: cls attn score with cls token (itself)
        l_idx, h_idx = metric_idx[:, left_tokens+1:], metric_idx[:, 1:left_tokens+1] # [B, N-K, 1], [B, K, 1]
        # split metric into two tokens and calculate cosine similarity
        B, N, D = metric.shape  # B, N, C//H
        metric = metric / metric.norm(dim=-1, keepdim=True)             # [B, N+1, C//H]
        lower = metric.gather(dim=-2, index=l_idx.expand(B, -1, D))     # [B, N-K, C//H]
        higher = metric.gather(dim=-2, index=h_idx.expand(B, -1, D))    # [B, K, C//H]
        scores = lower @ higher.transpose(-1, -2)                       # [B, N-K, K]

        scores[..., 0] = -math.inf    # delete score of cls token, [B, N-K, K]
        if distill_token:
            scores[..., :, 0] = -math.inf

        # get largest cosine similarity of each lower tokens
        node_max, node_idx = scores.max(dim=-1)                         # [B, N-K], [B, N-K]
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # [B, N-K, 1]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [B, N-K-r, 1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, [B, r, 1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # [B, r, 1]

        # if class_token:
        #     # Sort to ensure the class token is at the start
        #     unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        B, N, C = x.shape
        # get src (lower) and dst (higher) cls_attn score tokens
        unm = x.gather(dim=-2, index=unm_idx.expand(B, -1, C))  # [B, N-K-r, C]
        src = x.gather(dim=-2, index=src_idx.expand(B, -1, C))  # [B, r, C]
        dst = x.gather(dim=-2, index=dst_idx.expand(B, -1, C))  # [B, K, C]
        
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, -1, C), src, reduce=mode) # [B, K, C]

        if distill_token:
            return torch.cat([dst[:, :1], dst[:, 1:]], dim=1)
        else:
            return torch.cat((unm, dst),dim=1) # [B, N-r, C]
    
    return merge, h_idx


def merge_wavg_revised(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """

    x = merge(x * size, mode="sum") # [B, N+1-r, C]
    size = merge(size, mode="sum")  # [B, N+1-r, 1]

    x = x / size    # [B, N+1-r, C]
    return x, size


def merge_wavg_clsattn(
    merge: Callable, x: torch.Tensor, cls_attn: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    
    Arguments
        x: input tokens, [B, N+1, C]
        cls_attn: attention value with class token, [B, N+1, 1]
        size: size of tokens (how many tokens are merge), [B, N+1, 1]
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None]) # [B, N+1, 1]

    # custom 2': tome + weighted sum using attn score
    # x = merge(x * size * cls_attn, mode="sum")  # [B, N+1-r, C]
    # custom 3': tome + weighted sum using attn score
    x = merge(x * cls_attn, mode="sum")  # [B, N+1-r, C]
    cls_attn = merge(cls_attn, mode="sum")  # [B, N+1-r, 1]
    size = merge(size, mode="sum")    # [B, N+1-r, 1]

    x = x / cls_attn   # [B, N+1-r, C]
    return x, size


def merge_wavg_clsattn_evo(
    merge: Callable, x: torch.Tensor, cls_attn: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.

    custom 4'

    Arguments
        x: input tokens, [B, N+1, C]
        cls_attn: attention value with class token, [B, N+1, 1]
        size: size of tokens (how many tokens are merge), [B, N+1, 1]
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None]) # [B, N+1, 1]

    x = merge(x * cls_attn, mode="sum")     # [B, N+1-r, C]
    cls_attn = merge(cls_attn, mode="sum")  # [B, N+1-r, 1]
    size = merge(size, mode="sum")          # [B, N+1-r, 1]

    x = x / cls_attn            # [B, N+1-r, C]
    cls_attn = cls_attn / size  # [B, N+1-r, 1]
    return x, cls_attn, size


"""
이걸로 교체하기 전에 prop attn 옵션 껐는지 확인하기
"""
def merge_wavg_clsattn2(
    # merge: Callable, merge_idx: Callable, x: torch.Tensor, cls_attn: torch.Tensor, size: torch.Tensor = None
    merge: Callable, x: torch.Tensor, cls_attn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    
    Arguments
        x: input tokens, [B, N+1, C]
        cls_attn: attention value with class token, [B, N+1, 1]
    """

    x = merge(x * cls_attn, mode="sum")     # [B, N+1-r, C]
    cls_attn = merge(cls_attn, mode="sum")  # [B, N+1-r, 1]

    x = x / cls_attn
    return x


def prune_r(
    x: torch.Tensor, r: int, cls_attn: torch.Tensor, size: torch.Tensor = None
    ): 
    """
    Apply token pruning using attention score with cls token.
    Returns the pruned tensor and the new token sizes.

    Arguments
        x: input tokens, [B, N, C]
        cls_attn: attention value with class token, [B, N, 1]
        size: size of tokens (how many tokens are merge), [B, N, 1]
    """
    B, N, C = x.shape

    # We can only reduce by a maximum of 50% tokens
    r = min(r, N // 2)
    
    # +1 for cls token
    _, idx = torch.topk(cls_attn, N-r+1, dim=1, largest=True, sorted=True)  # [B, N-r+1, 1]
    index = idx.expand(B, N-r+1, C)                 # [B, N-r+1, C]
    x_pruned = torch.gather(x, dim=1, index=index)  # [B, N-r+1, C]
    size = torch.gather(size, dim=1, index=idx)     # [B, N-r+1, 1]
    
    return x_pruned, size


# ==================================================================================================
# ==================================================================================================
# ==================================================================================================

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None]) # [B, N+1, 1]

    x = merge(x * size, mode="sum") # [B, N+1-r, C]
    size = merge(size, mode="sum")  # [B, N+1-r, 1]

    x = x / size    # [B, N+1-r, C]
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

