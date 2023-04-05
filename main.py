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

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from datasets import build_dataset, build_dataloader

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import utils
import models
from arguments import add_arguments
from loggers import TensorboardLogger
from engine import train_one_epoch, evaluate
from losses import DistillLoss

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

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # make loggers
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # make datasets
    train_set, args.num_classes = build_dataset(args, is_train=True)
    val_set, _ = build_dataset(args, is_train=False)

    # make dataloaders
    train_loader, val_loader = build_dataloader(args, train_set, val_set, num_tasks, global_rank)
    cudnn.benchmark = True
    # mixup settings
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

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

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    # EMA (Exponential Moving Average) setting
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), and DP wrapper but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    # model distribute setting
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    total_batch_size = args.train_batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(train_set) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    if not args.eval:
        print("Number of training examples = %d" % len(train_set))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # hyperparmaeters setting
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        # have to fix this one
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # knowledge distillation setting
    teacher_model = None
    if args.distillation_type != 'none':
        # assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.num_classes,
            global_pool='avg',
        )
        # if args.teacher_path.startswith('https'):
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         args.teacher_path, map_location='cpu', check_hash=True)
        # else:
        #     checkpoint = torch.load(args.teacher_path, map_location='cpu')
        # teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

        # wrap the criterion in our custom DistillationLoss, which
        # just dispatches to the original criterion if args.distillation_type is 'none'
        criterion = DistillLoss(
            criterion, teacher_model, args.distillation_alpha, args.distillation_tau, print_mode=True)


    # resume model
    if args.resume:
        global_rank = utils.get_rank()
        checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(global_rank)))
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    # training
    loss_data, evalacc_data = [], []
    if not args.eval:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                args=args,
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                )

            lr_scheduler.step(epoch)

            eval_stats = evaluate(val_loader, model, device)
            print(f"Accuracy of the network on the {len(val_set)} eval images: {eval_stats['acc1']:.1f}%")

            is_best = False
            if max_accuracy < eval_stats["acc1"]:
                is_best = True
                max_accuracy = eval_stats["acc1"]
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'eval_{k}': v for k, v in eval_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.save_path and utils.is_main_process():
                save_path = Path(args.save_path)
                with (save_path / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if args.save_path:
                utils.save_on_master({
                    'args': args,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema else None,
                    'scaler': loss_scaler.state_dict(),
                }, args.save_path, is_best)

            loss_data.append(train_stats['loss'])
            evalacc_data.append(eval_stats['acc1'])
            

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        print(f'* (Acc) Max accuracy: {max_accuracy:.2f}%')

        # export loss, acc to excel
        acc_data = np.concatenate(([loss_data], [evalacc_data]), axis=0)
        utils.export_to_excel(args.save_path, acc_data)

    else:
        print('Start evaluation')
        start_time = time.time()
        eval_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the {len(val_set)} eval images: {eval_stats['acc1']:.1f}%")

        log_stats = {**{f'eval_{k}': v for k, v in eval_stats.items()},
                    'n_parameters': n_parameters}
            
        if args.save_path and utils.is_main_process():
            save_path = Path(args.save_path)
            with (save_path / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Eval time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)

