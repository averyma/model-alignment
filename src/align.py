import time
import numpy as np
import torch.nn as nn
from src.utils_log import Summary, AverageMeter, ProgressMeter
from src.evaluation import accuracy

def model_alignment(train_loader, source_model, witness_model, criterion_kd, criterion_cls, optimizer, lr_scheduler, scaler, device, args, is_main_task):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Model Alignment")

    len_history = len(train_loader) if not args.debug else 3
    loss_history = np.empty(len_history)
    loss_cls_history = np.empty(len_history)
    loss_align_history = np.empty(len_history)

    source_model.train()
    witness_model.eval()

    for param in witness_model.parameters():
        param.requires_grad = False

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        p_s = source_model(images)
        p_w = witness_model(images)

        loss_align = criterion_kd(p_s, p_w)

        loss_cls = criterion_cls(p_s, target)
        loss = args.lambda_kd * loss_align + args.lambda_cls * loss_cls

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(source_model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(source_model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step((i+1)/len(train_loader))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(p_s, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss_history[i] = loss.item()
        loss_cls_history[i] = loss_cls.item()
        loss_align_history[i] = loss_align.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)
        if args.debug and i == 2:
            break

    return top1.avg, top5.avg, losses.avg, [loss_history, loss_cls_history, loss_align_history]
