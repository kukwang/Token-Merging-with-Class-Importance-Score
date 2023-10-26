
# import os
# import sys
# import time
# import random
# import math
# import collections
import argparse
# import datetime
# import numpy as np
# import json

# from pathlib import Path

import torch
# import torch.backends.cudnn as cudnn
# from datasets import build_dataset, build_dataloader

# from timm.data.mixup import Mixup
from timm.models import create_model
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
# from timm.utils import NativeScaler, get_state_dict, ModelEma

import utils
import models
from arguments import add_arguments
import merge_module

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from merge_module.vis import make_visualization

def main(args):
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
    
    if args.reduce_num:
        print(f'reduce_num: {args.reduce_num}')
        merge_module.patch.timm(model)
        model.r = args.reduce_num
    else:
        print('no merge, no prune')


    # Source tracing is necessary for visualization!
    merge_module.patch.timm(model, trace_source=True)

    input_size = model.default_cfg["input_size"][1]

    # Make sure the transform is correct for your model!
    transform_list = [
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size)
    ]

    # The visualization and model need different transforms
    transform_vis  = transforms.Compose(transform_list)
    transform_norm = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

    img = Image.open(args.img_pth)
    img_vis = transform_vis(img)
    img_norm = transform_norm(img)

    # ============== merge =================
    model.r = args.reduce_num
    _ = model(img_norm[None, ...])
    source = model._tomecis_info["source"]

    print(f"{source.shape[1]} tokens at the end")
    vis = make_visualization(img_vis, source, patch_size=16, class_token=True)
    vis_name = f'vis_merge.jpg'
    vis.save(vis_name, 'JPEG')

    # # ============== merge up to a specific block =================
    # merge_block_num = 3
    # model.r = [13] * merge_block_num  # 6 / 12 layers
    # _ = model(img_norm[None, ...])
    # source = model._tomecis_info["source"]

    # print(f"{source.shape[1]} tokens at the end")
    # vis = make_visualization(img_vis, source, patch_size=16, class_token=True)
    # vis_name = f'vis_merge_stop_{merge_block_num}.jpg'
    # vis.save(vis_name, 'JPEG')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)