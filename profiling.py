# ===================================================================================
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ===================================================================================

import os
import sys
import time
import random
import math
import collections
import argparse
import datetime
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn

from timm.models import create_model
import torch.autograd.profiler as profiler

import utils
# import models
from arguments import add_arguments

import tome

def main(args):
    # device setting
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix random seeds
    print(f"random seed: {args.seed}")
    args.local_rank = utils.get_rank()
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed_all(args.seed + args.local_rank)
    np.random.seed(seed=args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cudnn.benchmark = True

    if args.mymodel:
        # create model
        model = create_model(
            args.model_name,
            pretrained=bool(args.pt_dl),
            num_classes=args.num_classes,
            drop_rate=args.dropout,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            model_dir=args.pt_dl,
            distillation_type=args.distillation_type
        )
        if args.pt_local is not None:
            print('start loading pretrained model from local')
            pretrained = torch.load(args.pt_local, map_location='cpu')
            pretrained = pretrained['model']
            utils.load_state_dict(model, pretrained)
        print('## model has been successfully loaded')

    else:
        # create model
        model = create_model(
            args.model_name,
            pretrained=bool(args.pt_dl),
            num_classes=args.num_classes,
            drop_rate=args.dropout,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        if args.pt_local is not None:
            print('start loading pretrained model from local')
            pretrained = torch.load(args.pt_local, map_location='cpu')
            pretrained = pretrained['model']
            utils.load_state_dict(model, pretrained)

    print('## model has been successfully loaded')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    model.to(device)


    input_size = (args.batch_size, 3, args.input_size, args.input_size)

    img = torch.randn(input_size).to(device)

    if args.tome_r:
        print(f'r: {args.tome_r}')
        if args.prune:
            prune_loc = (0)
            print(f'prune_loc: {prune_loc}')
            tome.patch.timm(model, prune_loc=prune_loc)
        else:
            tome.patch.timm(model)
        model.r = args.tome_r

    else:
        print('no merge, no prune')

    model(img)   # warmup
    
    with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
        model(img)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)

