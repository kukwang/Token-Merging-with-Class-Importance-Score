# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
import math

from typing import Tuple

import torch
# from timm.models.vision_transformer import Attention, Block, VisionTransformer
from models.vit import Attention, Block, VisionTransformer

import tome
import utils
from tome.merge import bipartite_soft_matching, bipartite_soft_matching_revised, merge_wavg, merge_source
from tome.utils import parse_r

# import torch.autograd.profiler as profiler

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        token_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None  # [B, N+1, 1]


        # # attention part
        # with profiler.record_function("attn"):
        # keep_rate = self.keep_rate
        # x_attn, metric = self.attn(self.norm1(x), token_size)   # tome, [B, N+1, C]
        x_attn, cls_attn, key, val = self.attn(self.norm1(x), token_size) # mine, tome + ATS weighted sum
        # x_attn, cls_attn, metric = self.attn(self.norm1(x), token_size) # custom 3'
        # x_attn, cls_attn, metric = self.attn(self.norm1(x), token_size, keep_rate)
        # x_attn, cls_attn, metric = self.attn(self.norm1(x), token_size, self._tome_info["cls_attn"]) # custom 4'
        # get reduce token number
        r = self._tome_info["r"]
        x = x + self._drop_path1(x_attn)    # [B, N+1, C]
        # if merge_flg:
        if r > 0:
            # if keep_rate < 1:
            # ============================================================================================================
            """
            tome
            """
            # # merge, _, top_r = bipartite_soft_matching(
            # merge, _ = bipartite_soft_matching(
            #     metric,
            #     r,
            #     class_token=self._tome_info["class_token"],
            #     distill_token=self._tome_info["distill_token"],
            # )

            # if self._tome_info["trace_source"]:
            #     self._tome_info["source"] = merge_source(
            #         merge, x, self._tome_info["source"]
            #     )

            # x, self._tome_info["size"] = merge_wavg(merge, x, token_size)
            # ============================================================================================================
            """
            mine: tome + weighted sum using significance score (ATS)
            """
            val_norm = torch.linalg.norm(val, ord=2, dim=2) # [B, N+1]
            score_ = cls_attn * val_norm[..., None] # calculate score used in weighted sum, [B, N+1, 1]

            merge = bipartite_soft_matching_revised(
                key,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )

            x, self._tome_info["size"] = tome.merge.merge_wavg_score(merge, x, score_, token_size)    # [B, K, C], [B, K, 1] 
            # ============================================================================================================
            """
            custom 4: split and prune & merge
            """
            # # split tokens and attention score based on attention map
            # B, N, C = x.shape
            # merge_rate = (1-keep_rate)/2
            # keep_rate = 1 - merge_rate

            # left_tokens = math.ceil(keep_rate * (N - 1))
            # merge_tokens = math.ceil(merge_rate * (N - 1))

            # # get sorted index of all tokens based on attn score
            # sorted_idx = cls_attn.argsort(dim=-2, descending=True)  # [B, N+1, 1], [B, N+1, 1]
            # # split token idxs into attentive and inattentive token idxs
            # attn_idx = sorted_idx[:, 1:left_tokens] # [B, K-1, 1]

            # # get attn and inattn tokens
            # attn_x = x.gather(dim=-2, index=attn_idx.expand(B, left_tokens-1, C))       # [B, K-1, C]

            # # get attn and inattn token's attention scores
            # metric_attn = metric.gather(dim=-2, index=attn_idx)     # [B, K, 1]

            # # bipartite soft matching attentive tokens
            # merge_attn = bipartite_soft_matching_revised(
            #     metric_attn,
            #     merge_tokens,
            #     self._tome_info["class_token"],
            #     self._tome_info["distill_token"],
            #     )

            # if token_size is None:
            #     token_size = torch.ones_like(x[..., 0, None]) # [B, N+1, 1]

            # # get token size of attn, inattn tokens
            # attn_size = token_size.gather(dim=-2, index=attn_idx)     # [B, K, 1]
            
            # # merge using weighted average
            # attn_x, attn_size = merge_wavg_revised(merge_attn, attn_x, attn_size)           # [B, K, C], [B, K, 1] 

            # # concat attn, inattn tokens and its sizes
            # x = torch.cat([x[:,0,None,:], attn_x], dim=1) # [B, N-K, C]
            # self._tome_info["size"] = torch.cat([token_size[:,0,None,:], attn_size], dim=1)  # [B, N+1-K, 1]
            # ============================================================================================================
            """
            custom 5,6: my evit
            """
            # # split tokens and attention score based on attention map
            # B, N, C = x.shape
            # left_tokens = math.ceil(keep_rate * (N - 1))

            # # get sorted index of all tokens based on attn score
            # sorted_idx = cls_attn.argsort(dim=-2, descending=True)  # [B, N+1, 1], [B, N+1, 1]
            # # split token idxs into attentive and inattentive token idxs
            # attn_idx = sorted_idx[:, 1:left_tokens] # [B, K-1, 1]

            # # get attn tokens
            # attn_x = x.gather(dim=-2, index=attn_idx.expand(B, left_tokens-1, C))       # [B, K-1, C]

            # # inattentive token fusion, custom 6
            # if self.fuse_token:
            #     compl = sorted_idx[:, left_tokens:] # [B, N+1-K, 1]
            #     non_topk = x.gather(dim=-2, index=compl.expand(-1, -1, C))  # [B, N+1-K, C]

            #     non_topk_attn = cls_attn.gather(dim=1, index=compl)  # [B, N+1-K, 1]
            #     extra_token = torch.sum(non_topk * non_topk_attn, dim=1, keepdim=True)  # [B, 1, C]
            #     # concat attn, inattn tokens and its sizes
            #     x = torch.cat([x[:,0,None,:], attn_x, extra_token], dim=1) # [B, N+1-K, C]
                
            # else:     # no token fusion, custom 5
            #     # concat attn, inattn tokens and its sizes
            #     x = torch.cat([x[:,0,None,:], attn_x], dim=1) # [B, N+1-K, C]
            # ============================================================================================================
            """
            custom 1', 2', 3': tome & weighted sum using attn score 
            """
            # merge = bipartite_soft_matching_revised(
            #     metric,
            #     r,
            #     self._tome_info["class_token"],
            #     self._tome_info["distill_token"],
            # )

            # # # custom 1': tome - prop attn + weighted sum using attn score
            # # x = merge_wavg_clsattn2(merge, x, cls_attn)           # [B, K, C]
            # # custom 2', 3': tome + weighted sum using attn score
            # # x, self._tome_info["size"] = tome.merge.merge_wavg_score(merge, x, cls_attn, token_size)    # [B, K, C], [B, K, 1] 
            # x, self._tome_info["size"] = tome.merge.merge_wavg_score(merge, x, score_, token_size)    # [B, K, C], [B, K, 1] 
            # ============================================================================================================
            """
            custom 4': tome + global attention evolution (Evo-ViT)
            """
            # merge = bipartite_soft_matching_revised(
            #     metric,
            #     r,
            #     self._tome_info["class_token"],
            #     self._tome_info["distill_token"],
            # )

            # # merge tokens and cls_attn
            # # return tokens, cls_attn and size, ([B, K, C], [B, K, 1], [B, K, 1])
            # x, self._tome_info["cls_attn"], self._tome_info["size"] = merge_wavg_clsattn_evo(merge, x, cls_attn, self._tome_info["size"]) 
            # ============================================================================================================
            """
            custom 12'~16': split by processed cosine similarity (sum, mean)
            """
            # merge = tome.merge.bipartite_soft_matching_sim(
            #     metric=metric,
            #     r=r,
            #     class_token=self._tome_info["class_token"],
            #     distill_token=self._tome_info["distill_token"],
            # )

            # # x, self._tome_info["size"] = merge_wavg(merge, x, token_size) # [B, K, C], [B, K, 1]
            # x, self._tome_info["size"] = tome.merge.merge_wavg_score(merge, x, cls_attn, token_size)    # [B, K, C], [B, K, 1] 
            # ============================================================================================================
            """
            custom 8': split and merge
            """
            # # split tokens and attention score based on attention map
            # B, N, C = x.shape   # B, N+1, C
            # if keep_rate < 1:   # if split stage,
            #     left_tokens = math.ceil(keep_rate * N-1)  # keep_rate * N
            #     # get sorted index of all tokens based on attn score
            #     cls_attn_sorted, sorted_idx = cls_attn.sort(dim=-2, descending=True)  # [B, N+1, 1], [B, N+1, 1]
            #     # split token idxs into attentive and inattentive token idxs
            #     attn_idx, inattn_idx = sorted_idx[:, 1:left_tokens+1], sorted_idx[:, left_tokens+1:] # [B, K, 1], [B, N-K, 1]

            #     # get attn and inattn tokens
            #     attn_x = x.gather(dim=-2, index=attn_idx.expand(B, left_tokens, C))       # [B, K, C]
            #     inattn_x = x.gather(dim=-2, index=inattn_idx.expand(B, N-1-left_tokens, C)) # [B, N-K, C]

            #     # get attn and inattn token's attention scores
            #     # metric_attn = metric.gather(dim=-2, index=attn_idx)     # [B, K, 1]
            #     metric_inattn = metric.gather(dim=-2, index=inattn_idx) # [B, N-K, 1]

            #     N_inattn = inattn_x.shape[1]
            #     attn_r, inattn_r = r, (N_inattn-1)//2

            #     # # bipartite soft matching attentive tokens
            #     # merge_attn = bipartite_soft_matching_revised(
            #     #     metric_attn,
            #     #     attn_r,
            #     #     class_token=False,
            #     #     distill_token=self._tome_info["distill_token"],
            #     #     )
            #     # bipartite soft matching inattentive tokens
            #     merge_inattn = bipartite_soft_matching_revised(
            #         metric_inattn,
            #         inattn_r,
            #         class_token=False,
            #         distill_token=self._tome_info["distill_token"],
            #         )

            #    # get token size of attn, inattn tokens
            #     attn_size = token_size.gather(dim=-2, index=attn_idx)     # [B, K, 1]
            #     inattn_size = token_size.gather(dim=-2, index=inattn_idx) # [B, N-K, 1]
                
            #     # split attention scores into attentive and inattentive token's attention scores, [B, K, 1], [B, N-K, 1]
            #     attn_cls_attn, inattn_cls_attn = cls_attn_sorted[:, 1:left_tokens+1], cls_attn_sorted[:, left_tokens+1:]
            #     # # weight average
            #     # attn_x, attn_size = merge_wavg_score(merge_attn, attn_x, attn_cls_attn, attn_size)           # [B, K, C], [B, K, 1] 
            #     # inattn_x, inattn_size = merge_wavg_score(merge_inattn, inattn_x, inattn_cls_attn, inattn_size) # [B, N-K, C], [B, N-K, 1]
            #     inattn_x, inattn_cls_attn, inattn_size = merge_wavg_clsattn_evo(
            #         merge_inattn, inattn_x, inattn_cls_attn, inattn_size) # custom 4'
                
            #     # concat attn, inattn tokens and its sizes
            #     x = torch.cat([x[:,0,None,:], attn_x, inattn_x], dim=1) # [B, N, C]
            #     self._tome_info["size"] = torch.cat([token_size[:,0,None,:], attn_size, inattn_size], dim=1)  # [B, N+1, 1]
            #     self._tome_info["cls_attn"] = torch.cat([cls_attn_sorted[:,0,None,:], attn_cls_attn, inattn_cls_attn], dim=1)  # [B, N+1, 1]
            # else:   # if not split stage,
            #     merge = bipartite_soft_matching_revised(
            #         metric,
            #         r,
            #         class_token=True,
            #         distill_token=self._tome_info["distill_token"],
            #     )

            #     # custom 3': tome + weighted sum using attn score
            #     # x, self._tome_info["size"] = merge_wavg_score(merge, x, cls_attn, self._tome_info["size"])    # [B, K, C], [B, K, 1] 
            #     x, self._tome_info["cls_attn"], self._tome_info["size"] = merge_wavg_clsattn_evo(
            #         merge, x, cls_attn, self._tome_info["size"]) # custom 4'
            # ============================================================================================================

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None    # tome
        # self, x: torch.Tensor, size: torch.Tensor = None, keep_rate: float = 1.0,
        # self, x: torch.Tensor, size: torch.Tensor = None, prev_cls_attn: torch.Tensor = None,    # custom 4', prev_cls_attn: [B, H, N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape   # B, N+1, C
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])  # make torchscript happy (cannot use tensor as tuple), [B, H, N+1, D]*3

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N+1, N+1]

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn_softmax = attn.softmax(dim=-1)     # [B, H, N+1, N+1]
        attn = self.attn_drop(attn_softmax)     # [B, H, N+1, N+1]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, N+1, C]
        x = self.proj(x)
        x = self.proj_drop(x)

        # if this block process pruning, get attn score with cls token 
        # if keep_rate < 1:
        # if True:    # custom 3': tome + weigthed sum
        cls_attn = attn[:, :, 0]                    # attn of cls token, [B, H, N+1]
        cls_attn = cls_attn.mean(dim=1)[..., None]  # mean over head, [B, N+1, 1]
        # return x, cls_attn, k.mean(1)
        return x, cls_attn, k.mean(1), v.mean(1)    # mine, [B, N+1, C], [B, N+1, 1], [B, N+1, D], [B, N+1, D]

        # # custom 4': tome + weighted sum + global attn evo (Evo-ViT)
        # if True:
        #     self.trade_off = 0.5
        #     cls_attn = attn[:, :, 0]                    # attn of cls token, [B, H, N+1]
        #     cls_attn = cls_attn.sum(dim=1)[..., None]   # sum over head, [B, N+1, 1]
        #     if prev_cls_attn is not None:
        #         cls_attn = (1-self.trade_off)*prev_cls_attn + self.trade_off*cls_attn   # global attention evolution [B, N+1, 1]
        #     return x, cls_attn, k.mean(1)   # [B, N+1, C], [B, N+1, 1], [B, N+1, N+1]

        # Return k as well here
        # return x, None, k.mean(1)
        return x, k.mean(1) # tome, [B, N+1, C], [B, N+1, D]


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = self.r
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            self._tome_info["cls_attn"] = None  # custom 4'

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, # tome
    # model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, trade_off: float = 0.5, # c4'
    # model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, base_keep_rate: float = 1.0, drop_loc: tuple = None,
    # model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, base_keep_rate: float = 1.0, drop_loc: tuple = None,
    #     trade_off: float = 0.5, # custom 10'
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    # model.keep_rate = 0.0
    model.r = 0

    # keep_rate = [1] * 12    # only deit-ti and deit-s!!!
    # if drop_loc is not None:
    #     for loc in drop_loc:
    #         keep_rate[loc] = base_keep_rate

    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    # # custom 4'
    # if trade_off < 1:
    #     model._tome_info["cls_attn"] = None

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    # block_cnt = 0
    # merge_flg = False
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
            # module.block_num = block_cnt
            # module.keep_rate = keep_rate[block_cnt]
            # block_cnt += 1
            # module.merge_flg = merge_flg
            # merge_flg = not merge_flg
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            # module.trade_off = trade_off      # custom 4'
