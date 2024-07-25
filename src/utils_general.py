import random
import os
import torch, torchvision
import torch.optim as optim
from typing import List, Optional, Tuple
import timm
import numpy as np

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(arch, device=None):

    if arch == 'vit_t_16':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    elif arch == 'vit_s_16':
        model = timm.create_model('vit_small_patch16_224', pretrained=False)
    elif arch == 'vit_b_16':
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
    elif arch == 'swin_s':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
    elif arch == 'swin_t':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    elif arch == 'inception_v3':
        # The ckpt for inv3 on torchvision only takes input size of 299x299,
        # whereas the ckpt from timm takes 256x256.
        model = timm.create_model('inception_v3', pretrained=False)
    else:
        # arch in ['resnet18', 'resnet50', 'resnet101', 'densenet121', 'vgg19_bn']
        model = torchvision.models.get_model(arch)

    if device is not None:
        model.to(device)

    return model

def get_optim(parameters, args):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    if args.optim.startswith("sgd"):
        opt = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    elif args.optim == "adamw":
        opt = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, AdamW are supported.")

    # check if milestone is an empty array
    if args.lr_scheduler_type == "multistep":
        _milestones = [args.epoch/ 2, args.epoch * 3 / 4]
        main_lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif args.lr_scheduler_type == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch - args.lr_warmup_epoch, eta_min=0.)
    elif args.lr_scheduler_type == "fixed":
        main_lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % args.lr_scheduler_type)

    if args.lr_warmup_epoch > 0:
        if args.lr_warmup_type == 'linear':
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                    opt, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        elif args.lr_warmup_type == 'constant':
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                    opt, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        else:
            raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epoch]
                )
    else:
        lr_scheduler = main_lr_scheduler

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

def ep2itr(epoch, loader):
    try:
        data_len = loader.dataset.data.shape[0]
    except AttributeError:
        data_len = loader.dataset.tensors[0].shape[0]
    batch_size = loader.batch_size
    iteration = epoch * np.ceil(data_len/batch_size)
    return iteration

def remove_module(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups
