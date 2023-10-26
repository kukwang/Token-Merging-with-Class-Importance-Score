# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
# import math

from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from merge_module.merge import bipartite_soft_matching_ToMeCIS, merge_wavg_score, merge_source
from merge_module.utils import parse_r


class ToMeCISBlock(Block):
    """
    Modifications:
     - Apply ToMe or ToMeCIS between the attention and mlp blocks
     - ToMe: Compute and propogate token size and potentially the token sources
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        token_size = self._tomecis_info["size"] if self._tomecis_info["prop_attn"] else None  # [B, N+1, 1]

        # attention part
        x_attn, cls_attn, key, val = self.attn(self.norm1(x), token_size)  # ToMeCIS, tome + ATS weighted sum

        r = self._tomecis_info["r"].pop(0)  # get reduce token number
        x = x + self._drop_path1(x_attn)    # [B, N+1, C]
        if r > 0:
            """
            ToMeCIS : Bipartite Soft Matching + weighted average with significance score
            """
            with torch.no_grad():
                val_norm = torch.linalg.norm(val, ord=2, dim=2) # [B, N+1]
                score_ = cls_attn * val_norm[..., None] # calculate score used in weighted average, [B, N+1, 1]

            merge = bipartite_soft_matching_ToMeCIS(
                key,
                r,
                self._tomecis_info["class_token"],
                self._tomecis_info["distill_token"],
            )
            if self._tomecis_info["trace_source"]:
                self._tomecis_info["source"] = merge_source(merge, x, self._tomecis_info["source"])

            x, self._tomecis_info["size"] = merge_wavg_score(merge, x, score_, token_size)    # [B, K, C], [B, K, 1] 
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - ToMe: Return the mean of k over heads from attention
     - ToMeCIS: Return the mean of k, v over heads from attention and first row from attention matrix 
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape   # B, N+1, C
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (qkv[0], qkv[1], qkv[2])  # make torchscript happy (cannot use tensor as tuple), [B, H, N+1, D]*3
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N+1, N+1], Q, K
        if size is not None:                # Apply proportional attention
            attn = attn + size.log()[:, None, None, :, 0]
        attn_softmax = attn.softmax(dim=-1)     # [B, H, N+1, N+1]
        attn = self.attn_drop(attn_softmax)     # [B, H, N+1, N+1]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, N+1, C]
        x = self.proj(x)    # [B, N+1, C]
        x = self.proj_drop(x)

        cls_attn = attn[:, :, 0].mean(dim=1)[..., None]  # mean attn of cls token over head, [B, 1, 1]
        return x, cls_attn, k.mean(1), v.mean(1)    # ToMeCIS, [B, N+1, C], [B, N+1, 1], [B, N+1, D], [B, N+1, D]

def make_tomecis_class(transformer_class):
    class ToMeCISVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tomecis_info["r"] = parse_r(len(self.blocks), self.r)
            self._tomecis_info["size"] = None
            self._tomecis_info["source"] = None
            
            return super().forward(*args, **kwdargs)

    return ToMeCISVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True,
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tomecis_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeCISVisionTransformer = make_tomecis_class(model.__class__)

    model.__class__ = ToMeCISVisionTransformer
    model.r = 0

    model._tomecis_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tomecis_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeCISBlock
            module._tomecis_info = model._tomecis_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
