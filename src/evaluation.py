import time
import torch
import torchattacks
from torch.utils.data import Subset
from src.context import ctx_noparamgrad_and_eval
from src.utils_log import Summary, AverageMeter, ProgressMeter

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def return_qualified(p_0, p_1, p_adv_0, p_adv_1, target):
    with torch.no_grad():
        _, pred_0 = p_0.topk(1, 1, True, True)
        _, pred_1 = p_1.topk(1, 1, True, True)
        _, pred_adv_0 = p_adv_0.topk(1, 1, True, True)
        _, pred_adv_1 = p_adv_1.topk(1, 1, True, True)

        pred_0 = pred_0.t()
        pred_1 = pred_1.t()
        pred_adv_0 = pred_adv_0.t()
        pred_adv_1 = pred_adv_1.t()

        correct_0 = pred_0.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_0 = pred_adv_0.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_1 = pred_adv_1.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        qualified = correct_0.eq(correct_1).eq(incorrect_0).eq(incorrect_1)

        return qualified

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        atk = get_attack(args.dataset, args.atk, model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if whitebox:
                with ctx_noparamgrad_and_eval(model):
                    delta = atk(images, target) - images
            else:
                delta = 0

            # compute output
            with torch.no_grad():
                output = model(images+delta)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_task:
                progress.display(i + 1)
            if args.debug:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ' if not whitebox else 'Whitebox: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

def get_attack(dataset, atk_method, model, eps, alpha, steps, random=True):
    if dataset in ['imagenet', 'food101', 'cars']:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if atk_method == 'pgd':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=random)
    elif atk_method == 'mi':
        attack = torchattacks.MIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'ni':
        attack = torchattacks.NIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'vni':
        attack = torchattacks.VNIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'vmi':
        attack = torchattacks.VMIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'sini':
        attack = torchattacks.SINIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'ti':
        attack = torchattacks.TIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_method == 'di':
        attack = torchattacks.DIFGSM(model, eps=eps, alpha=alpha, steps=steps)
    else:
        raise NotImplementedError('{} not supported'.format(atk_method))

    attack.set_normalization_used(mean=mean, std=std)
    return attack

def eval_transfer(val_loader, source_model, target_model, args, is_main_task):
    # Define total number of qualified samples to be evaluated
    num_eval = 100 if args.debug else 1000

    atk_source = get_attack(args.dataset, args.atk, source_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)
    atk_target = get_attack(args.dataset, args.atk, target_model, args.pgd_eps, args.pgd_alpha, args.pgd_itr if not args.debug else 1)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(source_model) and ctx_noparamgrad_and_eval(target_model):
            delta_source = atk_source(images, target) - images
            delta_target = atk_target(images, target) - images

        # compute output
        with torch.no_grad():
            p_source = source_model(images)
            p_target = target_model(images)
            p_adv_source = source_model(images+delta_source)
            p_adv_target = target_model(images+delta_target)
            qualified = return_qualified(p_source, p_target, p_adv_source, p_adv_target, target)

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_source2target = target_model((images+delta_source)[qualified, ::])

        acc1, acc5 = accuracy(p_source2target, target[qualified], topk=(1, 5))

        top1.update(acc1[0], num_qualified)
        top5.update(acc5[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1, top5, total_qualified],
        prefix='Transfer({}): '.format(args.atk))

    # switch to evaluate mode
    source_model.eval()
    target_model.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1.avg

