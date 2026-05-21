from pathlib import Path

import numpy as np
import math
import torch
import random
import os
import logging
import time


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(args, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        args.output_dir += '_test'
    if save_log:
        makedirs(args.output_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(args.log_dir + '/log.log')
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    args.logger = logger
    return args.logger


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(args, epoch, model, optimizer, projector=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        if projector:
            to_save = {
                'model': model.state_dict(),
                'projector': projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
        else:
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
        torch.save(to_save, checkpoint_path)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_weights(model_path, model):
    pretrained = torch.load(model_path)
    model.load_state_dict(pretrained, strict=False)
    del pretrained


def load_model(args, model, optimizer, projector=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'projector' in checkpoint and projector:
            projector.load_state_dict(checkpoint['projector'])
