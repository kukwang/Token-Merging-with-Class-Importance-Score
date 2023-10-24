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

import merge_module

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

    # create model
    model = create_model(
        args.model_name,
        pretrained=bool(args.pt_dl),
        num_classes=args.num_classes,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print('## model has been successfully loaded')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'model name: {args.model_name}')
    print('number of params:', n_parameters)

    model.to(device)

    runs = 500
    input_size = (3, args.input_size, args.input_size)

    if args.reduce_num:
        print(f'r: {args.reduce_num}')

        merge_module.patch.timm(model)
        model.r = args.reduce_num
    else:
        print('no merge')

    print(f'batch size: {args.batch_size}')
    tome_result = merge_module.utils.benchmark(
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

