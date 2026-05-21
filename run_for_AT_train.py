import os
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
import argparse

from IFKD.engine_for_AT_train import train_one_epoch, test
from IFKD.utils.datasets import get_loader
from IFKD.utils.projector import create_projector
from IFKD.utils.util import save_model, seed_torch, logger_configuration, load_model
from IFKD.models_AT import models_init_AT
from models import models_init
import torchvision.transforms as transforms


def get_args():
    parser = argparse.ArgumentParser(description='train')
    # train
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--save_model_freq', default=500, type=int)
    parser.add_argument('--test_model_freq', default=20, type=int)
    parser.add_argument('--print_step', default=10, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--global_step', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--seed', default=1024, type=int)

    # Dataset
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--train_data_path', default='/home/csudz/Desktop/dsm/dsm/Dataset/HR_Image_dataset', type=str)
    parser.add_argument('--test_data_path', default='/home/csudz/Desktop/dsm/dsm/Dataset/kodak_test', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)

    # model
    parser.add_argument('--model_s', type=str, default='vitjscc_small_C128', metavar='MODEL', )
    parser.add_argument('--model_t', type=str, default='vitjscc_large_C128', metavar='MODEL', )
    parser.add_argument('--model_at', type=str, default='vitjscc_AT_C128', metavar='MODEL', )
    parser.add_argument('--model_t_path',
                        default='/home/csudz/Desktop/dsm/file/dsm/project/IFKD/result/models/teacher/jscc_vit_large_snr7/checkpoint-1999.pth',
                        help='resume from checkpoint')
    parser.add_argument('--model_s_path',
                        default='/home/csudz/Desktop/dsm/file/dsm/project/IFKD/outputs/jscc_vit_small_C128_snr7/checkpoint-1999.pth',
                        help='resume from checkpoint')
    parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM', 'IBLoss'])

    # channel
    parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'])
    parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13')
    parser.add_argument('--given_snr', action='store_true')

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # output
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--log_dir', type=str, default='output')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device)
    seed_torch(args.seed)
    logger = logger_configuration(args, save_log=True)

    train_loader, test_loader = get_loader(args, args.num_workers)

    # teacher model and student model
    model_s = models_init.__dict__[args.model_s](channel_type=args.channel_type, multiple_snr=args.multiple_snr,
                                                 device=args.device)
    model_t = models_init.__dict__[args.model_t](channel_type=args.channel_type, multiple_snr=args.multiple_snr,
                                                 device=args.device)
    model_s.to(device)
    model_t.to(device)

    # projector
    data = torch.randn(2, 3, 256, 256).to(device)
    _ = model_s(data)
    _ = model_t(data)
    f_s = model_s.encoder.get_feature('enc_output')
    projector = create_projector(f_s.size(), f_s.size()).to(device)

    # assist teacher model
    f_s_list = model_s.encoder.get_feature('features')
    f_t_list = model_t.encoder.get_feature('features')
    f_s_shape = [f_s_list[i].shape for i in range(len(f_s_list))]
    f_t_shape = [f_t_list[i].shape for i in range(len(f_t_list))]
    model_a = models_init_AT.__dict__[args.model_at](channel_type=args.channel_type, multiple_snr=args.multiple_snr,
                                                     device=args.device, shape_s_list=f_s_shape, shape_t_list=f_t_shape)
    model_a.to(device)

    # load parameters
    checkpoint_t = torch.load(args.model_t_path, map_location='cpu',weights_only=False)
    checkpoint_model_t = checkpoint_t['model']
    model_t.load_state_dict(checkpoint_model_t)
    checkpoint_s = torch.load(args.model_s_path, map_location='cpu',weights_only=False)
    checkpoint_model_s = checkpoint_s['model']
    model_s.load_state_dict(checkpoint_model_s)

    # optimizer
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr
    cur_lr = args.lr
    model_params = [{'params': model_a.parameters(), 'lr': cur_lr}]
    optimizer = optim.Adam(model_params, lr=cur_lr)

    load_model(args, model_a, optimizer, projector)

    for _, p in model_t.named_parameters():
        p.requires_grad = False
    for _, p in model_s.named_parameters():
        p.requires_grad = False
    for _, p in model_a.named_parameters():
        p.requires_grad = True
    for _, p in projector.named_parameters():
        p.requires_grad = True

    if not args.eval:
        for epoch in tqdm(range(args.start_epoch, args.epochs), ncols=50):
            train_one_epoch(args, model_a, model_s, model_t, projector, train_loader, optimizer, epoch, logger)
            if (epoch + 1) % args.save_model_freq == 0:
                save_model(args, epoch, model_a, optimizer, projector)
    else:
        test(args, model_s, model_t, model_a, test_loader, logger)


if __name__ == '__main__':
    args = get_args()
    Path('./outputs').mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        args.output_dir = os.path.join('./outputs', args.output_dir)
        args.log_dir = os.path.join('./outputs', args.log_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
