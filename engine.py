# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time 
import torch
import time

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillLoss
import utils
import warnings
import loggers
from loggers import TensorboardLogger
# from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

def get_tau(start_tau, end_tau, ite, total):
    tau = start_tau + (end_tau - start_tau) * ite / total 
    return tau 

ite_step = 0
def train_one_epoch(args, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer: TensorboardLogger =None,
                    # writer: SummaryWriter =None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None):
    model.train(True)
    metric_logger = loggers.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # it_ = epoch * len(data_loader)
    
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // args.update_freq
        if step >= num_training_steps_per_epoch:
            continue

        if args.training_debug:
            if data_iter_step > args.training_debug:
                return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # it = start_steps + step  # global training iteration
        # # Update LR & WD for the first acc
        # if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % args.update_freq == 0:
        #     for i, param_group in enumerate(optimizer.param_groups):
        #         if epoch < param_group['fix_step']:
        #             param_group["lr"] = 0.
        #         elif lr_schedule_values is not None:
        #             param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
        #         if wd_schedule_values is not None and param_group["weight_decay"] > 0:
        #             param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            output = model(samples)
            if args.distillation_type != 'none':
                # DistillLoss
                loss, loss_part = criterion(samples, output, targets)
            else:
                # [SoftTargetCrossEntropy, LabelSmoothingCrossEntropy, CrossEntropyLoss]
                loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        # loss /= args.update_freq
        # loss.backward()
        # if (data_iter_step + 1) % args.update_freq == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     if model_ema is not None:
        #         model_ema.update(model)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # if mixup_fn is None:
        #     class_acc = (output.max(-1)[-1] == targets).float().mean()
        # else:
        #     class_acc = None
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_acc=class_acc)
        # min_lr = 10.
        # max_lr = 0.
        # for group in optimizer.param_groups:
        #     min_lr = min(min_lr, group["lr"])
        #     max_lr = max(max_lr, group["lr"])

        # metric_logger.update(lr=max_lr)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        # if writer is not None:
        if log_writer is not None:
            # writer.add_scalar("loss", loss_value, it)
            # writer.add_scalar("lr", optimizer.param_groups[0]["lr"], it)

            log_writer.update(loss=loss_value, head="loss")
            if args.distillation_type != 'none':
                log_writer.update(base_loss=loss_part[0], head="loss")
                log_writer.update(distill_loss=loss_part[1], head="loss")

            # log_writer.update(class_acc=class_acc, head="class_acc")
            # log_writer.update(lr=max_lr, head="opt")
            # log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = loggers.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, targets)
        else:
            output = model(images)
            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}