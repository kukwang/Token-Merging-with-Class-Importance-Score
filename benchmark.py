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

import utils
import models
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

    print(f'model name: {args.model_name}')
    print('number of params:', n_parameters)

    model.to(device)


    runs = 500
    input_size = (3, args.input_size, args.input_size)


    if args.tome_r:
        print(f'tome form')
        print(f'r: {args.tome_r}')

        tome.patch.timm(model)

        # if args.keep_rate < 1:
        #     drop_loc = eval(args.drop_loc)
        #     print(f'keep_rate: {args.keep_rate}')
        #     print(f'drop_loc: {drop_loc}')
        # if args.trade_off > 0:      # custom 4'
        #     print(f'tradeoff: {args.trade_off}')

        # tome.patch.timm(model,
        #                 # base_keep_rate=args.keep_rate,
        #                 # drop_loc=drop_loc,
        #                 trade_off=args.trade_off,       # custom 4'
        #                 )

        model.r = args.tome_r
        if args.threshold < 100:
            model.threshold = args.threshold
    else:
        print('no merge, no prune')

    print(f'batch size: {args.batch_size}')
    tome_result = tome.utils.benchmark(
        model,
        device=device,
        verbose=True,
        runs=runs,
        batch_size=args.batch_size,
        input_size=input_size
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)

