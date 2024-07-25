import os
import sys
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from distiller_zoo import DistillKL
from src.args import get_args, print_args
from src.utils_dataset import load_dataset, load_imagenet_test_shuffle
from src.utils_log import metaLogger, saveModel
from src.utils_general import seed_everything, get_model, get_optim, remove_module
from src.evaluation import validate, eval_transfer
from src.align import model_alignment

def ddp_setup(dist_backend, dist_url, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group(backend=dist_backend, world_size=world_size,
                            rank=rank, init_method=dist_url)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

def ddp_cleanup():
    dist.destroy_process_group()

def main():
    args = get_args()

    print_args(args)

    seed_everything(args.seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), start_method='spawn', join=True)
    else:
        # Simply call main_worker function
        args.gpu = 0 if torch.cuda.is_available() else None
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):

    args.ngpus_per_node = ngpus_per_node
    args.ncpus_per_node = len(os.sched_getaffinity(0))
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(args.gpu))

    assert (args.source_arch is None) != (args.eval_arch is None)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        ddp_setup(args.dist_backend, args.dist_url, args.rank, args.world_size)
        dist.barrier()

    # Load source model/eval model
    source_model = get_model(args.source_arch if args.eval_arch is None else args.eval_arch)
    ckpt = torch.load(args.source_dir if args.eval_arch is None else args.eval_dir, map_location=device)
    try:
        source_model.load_state_dict(ckpt)
    except RuntimeError:
        source_model.load_state_dict(remove_module(ckpt))
    print('{}: Load {} model from {}.'.format(device,
        'source' if args.eval_arch is None else 'eval',
        args.source_dir if args.eval_arch is None else args.eval_dir))

    # Load witness model
    if args.eval_arch is None:
        witness_model = get_model(args.witness_arch)
        ckpt = torch.load(args.witness_dir, map_location=device)
        try:
            witness_model.load_state_dict(ckpt)
        except RuntimeError:
            witness_model.load_state_dict(remove_module(ckpt))
        print('{}: Load witness model from {}.'.format(device, args.witness_dir))
    else:
        print('Evaluation ONLY! Skip loading witness model.')

    result = {'loss': None,
              'loss_cls': None,
              'loss_align': None,
              'test-err': None,
              'whitebox-err': None,
              args.target_arch: None}

    # Sending the model to the device
    if not torch.cuda.is_available():
        print('This should not be run on CPU!!!!!')
        return 0
    elif args.distributed:
        # Compute batch size and workers for distributed training
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = args.ncpus_per_node//max(args.ngpus_per_node, 1)
        print("GPU: {}, batch_size: {}, ncpus_per_node: {},"
              "ngpus_per_node: {}, workers: {}".format(
                  args.gpu, args.batch_size, args.ncpus_per_node,
                  args.ngpus_per_node, args.workers))

        torch.cuda.set_device(args.gpu)
        source_model.cuda(args.gpu)
        source_model = DDP(source_model, device_ids=[args.gpu])
        if args.eval_arch is None:
            witness_model.cuda(args.gpu)
    else:
        torch.cuda.set_device(args.gpu)
        source_model = source_model.cuda(args.gpu)
        if args.eval_arch is None:
            witness_model = witness_model.cuda(args.gpu)

    # Set the main task for the main process
    is_main_task = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    print('{}: is_main_task: {}'.format(device, is_main_task))

    # Define the loss function: cls for classification, kd for alignment
    criterion_cls = nn.CrossEntropyLoss().to(device)
    if args.eval_arch is None:
        criterion_kd = DistillKL(args.kl_temp).to(device)

        # Define the optimizer and learning rate scheduler
        opt, lr_scheduler = get_optim(source_model.parameters(), args)
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        if is_main_task:
            print('{}: agrs.amp: {}, scaler: {}'.format(device, args.amp, scaler))

    if args.distributed:
        dist.barrier()

    # Create loggers
    if is_main_task:
        print('{}: This is the device for the main task!'.format(device))
        print('{}: local logger created!'.format(device))
        logger = metaLogger(args)
        logging.basicConfig(
            filename=args.output_dir+ "/log/log.txt",
            format='%(asctime)s %(message)s', level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # train_loader and test_loader are the original loader for imagenet
    train_loader, test_loader, train_sampler, _ = load_dataset(args.dataset,
                                                               args.data_dir,
                                                               args.batch_size,
                                                               args.workers,
                                                               args.distributed)

    test_loader_shuffle, val_sampler = load_imagenet_test_shuffle(args.data_dir,
                                                                  batch_size=32,
                                                                  workers=0,
                                                                  distributed=args.distributed)

    print('{}: Dataloader compelete!'.format(device))

    if is_main_task and args.eval_arch is None:
        print('Modifying {} with {} using {}!'.format(args.source_arch, args.witness_arch, args.method))
##########################################################
###################### Alignment begins ##################
##########################################################
    if args.eval_arch is None:
        if args.distributed:
            dist.barrier()
            train_sampler.set_epoch(27)
        train_acc1, train_acc5, loss, loss_history = model_alignment(train_loader,
                                                                     source_model,
                                                                     witness_model,
                                                                     criterion_kd,
                                                                     criterion_cls,
                                                                     opt,
                                                                     lr_scheduler,
                                                                     scaler,
                                                                     device,
                                                                     args,
                                                                     is_main_task)
        # checkpointing for preemption
        if is_main_task:
            if args.save_modified_model:
                print('Saving aligned model!')
                saveModel(args.output_dir+"/model/", "final_model", source_model.state_dict())

            result['loss'] = loss
            result['loss_cls'] = loss_history[1].mean()
            result['loss_align'] = loss_history[2].mean()

        if args.distributed:
            dist.barrier()

        del train_loader
        del witness_model
        torch.cuda.empty_cache()
##########################################################
###################### Alignment Ends ####################
##########################################################
###################### Evaluation Begins #################
##########################################################

    # Model error on the unperturbed test data
    if args.eval_standard:
        if args.distributed:
            dist.barrier()
        test_acc1, _ = validate(test_loader, source_model, criterion_cls, args, is_main_task)
        test_err1 = 100.-test_acc1
        if is_main_task:
            print(' *  {}: {:.2f}\n *  {}: {:.2f}'.format('test-err', test_err1))

    # Model error under the whitebox attack
    if args.eval_whitebox:
        if args.distributed:
            dist.barrier()
        whitebox_acc1, _ = validate(test_loader_shuffle, source_model, criterion_cls, args, is_main_task, whitebox=True)
        whitebox_err1 = 100.-whitebox_acc1
        if is_main_task:
            print(' *  {}: {:.2f}\n *  {}: {:.2f}'.format('whitebox-err', whitebox_err1))

    if args.distributed:
        dist.barrier()

    # Load target model
    target_model = get_model(args.target_arch)
    ckpt = torch.load(args.target_dir, map_location=device)
    try:
        target_model.load_state_dict(ckpt)
    except RuntimeError:
        target_model.load_state_dict(remove_module(ckpt))
    print('{}: Load target model from {}.'.format(device, args.target_dir))
    target_model.cuda(args.gpu)
    if args.distributed:
        target_model = DDP(target_model, device_ids=[args.gpu])
        dist.barrier()
        val_sampler.set_epoch(27)

    # Evaluate transferability (from source to target)
    acc1_transfer = eval_transfer(test_loader_shuffle, source_model, target_model, args, is_main_task)
    err1_transfer = 100.-acc1_transfer
    result[args.target_arch] = err1_transfer

    if args.distributed:
        dist.barrier()
    if is_main_task:
        print(' *  {}: {:.2f}'.format(args.target_arch, err1_transfer))
##########################################################
###################### Evaluation Ends ###################
##########################################################
        # Logging and checkpointing only at the main task (rank0)
        print('result: {}'.format(result))

        for key in result.keys():
            if result[key] is not None:
                logger.add_scalar(key, result[key], 1)
                logging.info("{}: {:.2f}\t".format(key, result[key]))
        logger.save_log(is_final_result=True)

    if args.distributed:
        dist.barrier()
        ddp_cleanup()

if __name__ == "__main__":
    main()
