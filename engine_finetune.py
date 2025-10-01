# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def compute_per_class_accuracy(all_preds, all_targets, num_classes=10):
    """
    Compute per-class accuracy
    
    Args:
        all_preds: tensor of shape [N] with predicted class indices
        all_targets: tensor of shape [N] with ground truth class indices
        num_classes: number of classes
    
    Returns:
        per_class_acc: dict with class_id -> accuracy mapping
    """
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    for pred, target in zip(all_preds, all_targets):
        per_class_total[target] += 1
        if pred == target:
            per_class_correct[target] += 1
    
    per_class_acc = {}
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_acc[i] = (per_class_correct[i] / per_class_total[i]).item() * 100
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc, per_class_correct, per_class_total

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, compute_per_class=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_preds = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # Store predictions for per-class accuracy
        if compute_per_class:
            _, pred = output.topk(1, 1, True, True)
            all_preds.append(pred.squeeze(-1))
            all_targets.append(target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if compute_per_class:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        per_class_acc, per_class_correct, per_class_total = compute_per_class_accuracy(
            all_preds, all_targets
        )

        # Sort by class index
        for class_idx in sorted(per_class_acc.keys()):
            print(f"  Class {class_idx:4d}: {per_class_acc[class_idx]:6.2f}% "
                f"({int(per_class_correct[class_idx]):5d}/{int(per_class_total[class_idx]):5d})")

        # Print summary statistics
        accuracies = list(per_class_acc.values())
        print(f"\nPer-class accuracy summary:")
        print(f"  Mean: {sum(accuracies)/len(accuracies):.2f}%")
        print(f"  Min:  {min(accuracies):.2f}%")
        print(f"  Max:  {max(accuracies):.2f}%")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}