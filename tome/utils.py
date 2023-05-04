# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0   # number of processed images
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
    if batch_size > 1:
        return throughput
    else:
        latency = elapsed / total
        if verbose:
            print(f"latency: {latency:.5f} s/im")
        return throughput, latency


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

def get_unique_indices(
    indices: torch.Tensor,  # picked index list, [B, N]
    max_value: int          # max value of index, N-1
    ) -> torch.Tensor:
    """
    :param indices: indices of the tokens to be sampled
    :param max_value: maximum number of the tokens to be sampled
    :return: unique indices of the tokens to be sampled
    """
    sorted_indices = indices.sort(dim=1)[0]  # sort idx, get index value, [B, N]

    shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)    # shift left (cls token), [B, N]
    unique_indices = torch.where(
        condition=(shift_left - sorted_indices) == 0,
        input=max_value * torch.ones_like(indices),
        other=sorted_indices,
    )   # get unique indices, make max val if already exists, [B, N]
    unique_indices = unique_indices.sort(dim=1)[0]   # sort unique indices, [B, N]
    return unique_indices   # [B, N]


def create_ys(
    normalized_cdf: torch.Tensor,   # [B, N]
    n_tokens: int                   # N
    ) -> torch.Tensor:
    """
    Sample uniformly from y-axis.
    """

    B = normalized_cdf.shape[0]     # B
    # epsilon = (1 / (n_tokens - 1)) / 2
    # get uniformely stepped point
    ys = (torch.linspace(start=0,end=1.0,steps=n_tokens,device=normalized_cdf.device,
                            ).unsqueeze(0).repeat(B, 1))   # [B, N]

    # get start point of ys using min of normalized cdf, ignore zeros
    ys_start = (torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1
                            )[0].unsqueeze(-1).expand_as(ys)) # [B, N]

    # get steps of ys
    steps = (torch.range(0, n_tokens - 1, device=normalized_cdf.device
                            ).unsqueeze(0).expand_as(ys_start))   # [B, N]

    # get ys that uniformely increasing value from minimum val to 1
    ys = ys_start + (((ys * (n_tokens - 1)) - ys_start * steps) / (n_tokens - 1))   # [B, N]

    return ys   # [B, N]


def inverse_transform_sampling(
    sorted_scores: torch.Tensor,    # sorted token score, ascending, [B, N]
    sorted_indices: torch.Tensor,   # sorted token score indics, ascending, [B, N]
    sim: torch.Tensor,             # attention value, [B, N]
    ):
    """
    Sample tokens based on their similarity
    """
    B, N = sim.shape   # B, N

    cdf = sorted_scores.cumsum(dim=1)  # get CDF of scores (scores treated as probs), [B, N]
    # normalized cdf, [B, N]
    normalized_cdf = (cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

    # sampled values from y-axis (unoformly increase minimum to 1)
    ys = create_ys(normalized_cdf, N).unsqueeze(dim=2)      # [B, N, 1]
    normalized_cdf = normalized_cdf.unsqueeze(dim=1)        # [B, 1, N]

    expanded_ys = ys.expand(B, ys.shape[1], ys.shape[1])    # [B, N, N]

    # get min indices of ys, if dim is not matched with cdf, add padding to the front
    tokens_to_pick_ind = torch.abs(expanded_ys - normalized_cdf).min(dim=2)[1]  # [B, N]
    print(f'tokens_to_pick_ind shape: {tokens_to_pick_ind.shape}')
    # get sorted sim matrix depending on similarity
    sim_sorted = sim.gather(
        dim=-1,
        index=sorted_indices,
    )  # [B, N]

    # get unique indices
    unique_indices = get_unique_indices(indices=tokens_to_pick_ind, max_value=N-1)[:, :N-1]   # [B, N-1]

    # Prune the attention matrix
    sim = sim_sorted.gather(
        dim=-1,
        index=unique_indices,
        )   # [B, N]

    mask = (unique_indices != (N - 1)).unsqueeze(-1).float()  # update mask, [B, N-1, 1]

    sampled = torch.nonzero(mask)     # sampled tokens, nonzero of mask, [N'', 3]
    print(sampled.shape)
    return sim, mask, sampled # [B, N], [B, N-1, 1], [N'', 3]

# ======================================================================================================
# ======================================================================================================

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
