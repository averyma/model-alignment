'''
mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
'''
import os
import torch
import numpy as np
import torch.utils.data.distributed
from torchvision import datasets, transforms


def load_dataset(dataset, data_dir, batch_size=128, workers=4, distributed=False):

    # default augmentation
    if dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list = [transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]
        transform_train = transforms.Compose(transform_list)

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    # load dataset
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    data_train = datasets.ImageFolder(traindir, transform_train)
    data_test = datasets.ImageFolder(valdir, transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test, shuffle=False, drop_last=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(data_train)
        val_sampler = torch.utils.data.SequentialSampler(data_test)

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, test_loader, train_sampler, val_sampler

def load_imagenet_test_shuffle(data_dir, batch_size=128, workers=4, distributed=False):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # load dataset
    valdir = os.path.join(data_dir, 'val')
    data_test = datasets.ImageFolder(valdir, transform_test)

    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test)
    else:
        val_sampler = torch.utils.data.RandomSampler(data_test)

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=(val_sampler is None),
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return test_loader, val_sampler
