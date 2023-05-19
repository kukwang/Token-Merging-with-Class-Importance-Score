# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import math
import time
import numpy as np
from itertools import repeat
import collections.abc
from collections import deque
from typing import Any, List
from numbers import Number
from pathlib import Path
import pandas as pd

import torch
import torch.distributed as dist
from torch._six import inf
from timm.utils import get_state_dict
from fvcore.nn import FlopCountAnalysis


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# ====================================================================================
# device setting
# ====================================================================================
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state_dict, pth, is_best):
    if is_main_process():
        # save last ep
        torch.save(state_dict, os.path.join(pth, 'checkpoint_ep{}.pth.tar'.format(state_dict['epoch'])))
        if os.path.isfile(os.path.join(pth, 'checkpoint_ep{}.pth.tar'.format(state_dict['epoch'] - 1))):
            os.remove(os.path.join(pth, 'checkpoint_ep{}.pth.tar'.format(state_dict['epoch'] - 1)))

        if is_best:
            file_list = os.listdir(pth)

            # save winner
            torch.save(state_dict, os.path.join(pth, 'best_acc_ep{}.pth.tar'.format(state_dict['epoch'])))
            # remove last winner (both pth and excel)
            best_file_list = [file for file in file_list if file.startswith("best_acc_ep")]
            for name in best_file_list:
                epoch_num = int(name.split('.')[0].split('ep')[-1])
                if epoch_num < state_dict['epoch']:
                    os.remove(os.path.join(pth, name))


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# ====================================================================================
# load & save model
# ====================================================================================
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()

    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def load_state_dict_copy(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()

    for state in list(state_dict.keys()):
        if 'attn' in state:
            loc = state.find('qkv')
            if loc > 0:
                loc += 3
            else:
                loc = state.find('proj') + 4
            name = state[:loc] + 'copy' + state[loc:]
            state_dict[name] = state_dict[state]
    # for state in state_dict:
    #     if 'qkv' in state:
    #         state_dict[state] = state_dict.pop(state)

    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model, model_without_ddp, optimizer, best_acc, best_acc_ema, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
            'best': best_acc,
            'best_ema': best_acc_ema
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            print("With optim & sched!")
        if 'best' in checkpoint:
            max_accuracy = checkpoint['best']
            print('Previous best accuracy: %.2f' % max_accuracy)
        if 'best_ema' in checkpoint:
            max_accuracy_ema = checkpoint['best_ema']
            print('Previous best accuracy_ema: %.2f' % max_accuracy_ema)
        return max_accuracy, max_accuracy_ema
    else:
        return [0., 0.]


def export_to_excel(pth, data, tag=''):
    df = pd.DataFrame(data)
    df = (df.T)
    # df.columns = ['loss', 'val_acc']
    if tag == '':
        df.columns = ['acc', 'avg_sim']
    elif tag == 'class':
        df.columns = ['score_diff_std', 'score_diff_avg']

    xlsx_path = pth + '/acc_log' + tag + '.xlsx'

    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:  # pip install xlsxwriter
        df.to_excel(writer)
    print("SAVE " + xlsx_path + " successfully")
    writer.close()

# ====================================================================================
# helper functions
# ====================================================================================
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

        
def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # pdas a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n


# ====================================================================================
# evaluate the model's performance
# ====================================================================================
def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops


def calc_flops(model, img_size=224, show_details=False, ratios=None):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size)
        model.default_ratio = ratios
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("#### GFLOPs: {} for ratio {}".format(flops1 / 1e9, ratios))
    return flops1 / 1e9


@torch.no_grad()
def throughput(images, model):
    model.eval()

    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)