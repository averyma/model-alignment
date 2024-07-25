import yaml
import argparse
import os
import distutils.util

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method",
    			default=argparse.SUPPRESS, type=str)
    parser.add_argument("--dataset",
    			default=argparse.SUPPRESS, type=str)

    # hyper-param for optimization
    parser.add_argument("--lr",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--lr_scheduler_type",
    			default=argparse.SUPPRESS, type=str)
    parser.add_argument("--optim",
    			default=argparse.SUPPRESS, type=str)
    parser.add_argument("--momentum",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--weight_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--nesterov",
                default=argparse.SUPPRESS, type=distutils.util.strtobool)
    parser.add_argument("--lr_warmup_type",
    			default=argparse.SUPPRESS, type=str)
    parser.add_argument("--lr_warmup_epoch",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--lr_warmup_decay",
    			default=argparse.SUPPRESS, type=float)
    parser.add_argument("--batch_size",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--epoch",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument("--seed",
    			default=argparse.SUPPRESS, type=int)
    parser.add_argument('--amp',
    			default=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--clip_grad_norm',
                default=None)
    parser.add_argument("--save_modified_model",
                default=False, type=distutils.util.strtobool)
    parser.add_argument("--kl_temp",
                default=4., type=float)
    parser.add_argument("--lambda_cls",
                default=0., type=float)
    parser.add_argument("--lambda_kd",
                default=1., type=float)

    parser.add_argument("--output_dir",
                        default='str', required=True)
    parser.add_argument("--data_dir",
                        default='str', required=True)

    # single-node multi-gpu setup
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
    parser.add_argument('--debug',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument("--print_freq",
                        default=argparse.SUPPRESS, type=int)


    # model alignment/misalignment
    parser.add_argument("--source_arch",
    			default=None, type=str)
    parser.add_argument("--source_dir",
    			default=None, type=str)
    parser.add_argument("--target_arch",
    			default=None, type=str)
    parser.add_argument("--target_dir",
    			default=None, type=str)
    parser.add_argument("--witness_arch",
    			default=None, type=str)
    parser.add_argument("--witness_dir",
    			default=None, type=str)
    parser.add_argument("--eval_arch",
    			default=None, type=str)
    parser.add_argument("--eval_dir",
    			default=None, type=str)
    parser.add_argument('--eval_standard',
                        default=False, type=distutils.util.strtobool)
    parser.add_argument('--eval_whitebox',
                        default=False, type=distutils.util.strtobool)

    parser.add_argument("--pgd_itr",
                            default=20, type=int)
    parser.add_argument("--pgd_eps",
                            default=4./255., type=float)
    parser.add_argument("--pgd_alpha",
                            default=1./255., type=float)
    parser.add_argument("--atk",
                            default='pgd', type=str)

    args = parser.parse_args()

    return args

def make_dir(args):
    config_dir = str(args["output_dir"]+"/config/")
    log_dir = str(args["output_dir"]+"/log/")
    model_dir = str(args["output_dir"]+"/model/")

    try:
        os.makedirs(config_dir)
        os.makedirs(log_dir)
        os.makedirs(model_dir)
    except os.error:
        pass

    if not os.path.exists(config_dir + "/config.yaml"):
        f = open(config_dir + "/config.yaml" ,"w+")
        f.write(yaml.dump(args))
        f.close()

def get_default(yaml_path):
    default = {}
    with open(yaml_path, 'r') as handle:
        default = yaml.load(handle, Loader=yaml.FullLoader)
    return default

def get_base_model_dir(yaml_path):
    with open(yaml_path, 'r') as file:
        base_model_dir = yaml.safe_load(file)
    return base_model_dir

def get_args():
    args = parse_args()
    default = get_default('options/default.yaml')

    default.update(vars(args).items())

    make_dir(default)

    if default['clip_grad_norm'] == None:
        pass
    elif default['clip_grad_norm'] in ['none', 'None']:
        default['clip_grad_norm'] = None
    else:
        default['clip_grad_norm'] = float(default['clip_grad_norm'])

    return argparse.Namespace(**default)

def print_args(args):
    print("***********************************************************")
    print("************************ Arguments ************************")
    print("***********************************************************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("***********************************************************")
    print("***********************************************************")
