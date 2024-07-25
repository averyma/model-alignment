import time
import os
import wandb
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from enum import Enum
import torch.distributed as dist

def saveModel(save_path, model_name, model_state_dict):
    if not os.path.exists(save_path):
        print("Specified path ({}) does not exist, so nothing got saved.".format(save_path))
    else:
        model_path = save_path+model_name+'.pt'
        load_successfully = False
        save_counter = 0
        while not load_successfully and save_counter < 10:
            torch.save(model_state_dict, model_path)
            try:
                torch.load(model_path)
            except:
                print('Failed to load model!')
                save_counter += 1
                print('{}th attempt at saving model'.format(save_counter))
            else:
                print('Model saved and verified!')
                load_successfully = True

class metaLogger(object):
    def __init__(self, args):
        self.log_path = args.output_dir+"/log/"
        self.ckpt_status = self.get_ckpt_status(args.output_dir)
        self.log_dict = self.load_log(self.log_path)

    def get_ckpt_status(self, output_dir):
        ckpt_dir = os.path.join(output_dir, 'ckpt')
        ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")
        ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")

        if not os.path.exists(ckpt_location_curr) and not os.path.exists(ckpt_location_prev):
            return 'none' # this results in an invalid ckpt_location

        try:
            print('[metaLogger]: getting ckpt_curr status')
            torch.load(ckpt_location_curr)
        except:
            print('failed to load ckpt_curr')
        else:
            return "curr"

        try:
            print('[metaLogger]: getting ckpt_prev status')
            torch.load(ckpt_location_prev)
        except:
            print('failed to load ckpt_prev')
        else:
            return "prev"

        print('ckpt_curr & ckpt_prev exist, but both failed to load')

        return 'corrupted'


    def load_log(self, log_path):
        if self.ckpt_status == "curr":
            log_dict = torch.load(log_path + "/log_curr.pth")
        elif self.ckpt_status == 'prev':
            log_dict = torch.load(log_path + "/log_prev.pth")
        else:
            log_dict = defaultdict(lambda: list())

        return log_dict

    def add_scalar(self, name, val, step):
        try:
            self.log_dict[name] += [(time.time(), int(step), float(val))]
        except KeyError:
            self.log_dict[name] = [(time.time(), int(step), float(val))]

    def add_scalars(self, name, val_dict, step):
        # self.writer.add_scalars(name, val_dict, step)
        for key, val in val_dict.items():
            self.log_dict[name+key] += [(time.time(), int(step), float(val))]

    def add_figure(self, name, val):
        path = self.log_path + "/" + name + ".png"
        val.savefig(path)

    def save_log(self, is_final_result=False):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass

        if not is_final_result:
            log_prev = os.path.join(self.log_path, "log_prev.pth")
            log_curr = os.path.join(self.log_path, "log_curr.pth")

            # no existing logs
            if not (os.path.exists(log_curr) or os.path.exists(log_prev)):
                pass
            elif os.path.exists(log_curr):
                # overwrite log_prev with log_curr
                cmd = "cp -r {} {}".format(log_curr, log_prev)
                os.system(cmd)

        log_final = os.path.join(self.log_path, "log_final.pth")
        saved_path = log_final if is_final_result else log_curr

        load_successfully = False
        save_counter = 0
        while not load_successfully and save_counter < 10:
            torch.save(dict(self.log_dict), saved_path)
            try:
                torch.load(saved_path)
            except:
                print('Failed to load log!')
                save_counter += 1
                print('{}th attempt at saving log'.format(save_counter))
            else:
                print('Log saved and verified!')
                load_successfully = True

class wandbLogger(object):
    def __init__(self, args):
        self.log_path = args.output_dir+"/log/"
        self.wandb_log = wandb.init(name=args.output_dir.split("/")[-1], project=args.wandb_project,
                                    dir=args.output_dir,
                                    resume=True,
                                    reinit=True)
        self.wandb_log.config.update(args, allow_val_change=True)

    def upload(self, logger, epoch):
        all_keys = [*logger.log_dict.keys()]
        for _epoch in range(epoch):
            commit = ((_epoch+1)==epoch)
            wandb_dict = OrderedDict()
            wandb_dict['epoch'] = _epoch+1

            for key in all_keys:
                if (_epoch+1) in np.array(logger.log_dict[key])[:, 1]:
                    idx = (_epoch+1) == np.array(logger.log_dict[key])[:, 1]
                    wandb_dict[key] = np.array(logger.log_dict[key])[idx, 2]
            self.wandb_log.log(wandb_dict, step=_epoch+1, commit=commit)

    def finish(self):
        wandb.finish()

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

