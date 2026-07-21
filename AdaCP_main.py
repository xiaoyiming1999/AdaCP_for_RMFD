#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import logging
import warnings
import numpy as np
import torch

from utils.logger import setlogger
from utils.AdaCP_train import train_AdaCP_utils
from utils.Vanilla_train import train_Vanilla_utils

warnings.filterwarnings('ignore')

def seed_torch(seed=120):
    seed = int(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# seed_torch(seed=122)

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model parameters
    parser.add_argument('--model_name', type=str, default='Adaptive CP', help='the name of the model')
    parser.add_argument('--model', type=str, default='Resnet18',
                        choices=['Resnet18', 'WideResnet', 'MobileNet', 'DRSN', 'vgg11'])
    parser.add_argument('--method', type=str, default='Ada_cp',
                        choices=['Vanilla', 'label_smooth', 'focal_loss', 'confidence', 'Ada_cp', 'LogNorm'])

    # data parameters
    parser.add_argument('--data_name', type=str, default='BJUT', choices=['MCC5', 'BJUT', 'THU', 'HUST', 'HNU'])
    parser.add_argument('--data_length', type=int, default=1024)
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--classes', type=list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--op_conditions', type=list, default=[0])
    parser.add_argument('--noise_SNR', type=int, default=100)
    parser.add_argument('--num_train_samples', type=int, default=400)
    parser.add_argument('--num_val_samples', type=int, default=100)
    parser.add_argument('--num_test_samples', type=int, default=100)

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 Regularization')
    parser.add_argument('--initial_trade', type=float, default=0.1)
    parser.add_argument('--lamda', type=float, default=1.5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='15,30', help='the learning rate decay for step and stepLR')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--num_bins', type=int, default=10)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.method == 'Ada_cp':
        trainer = train_AdaCP_utils(args, save_dir)
    elif args.method == 'Vanilla' or args.method == 'focal_loss' or args.method == 'confidence' \
            or args.method == 'label_smooth' or args.method == 'LogNorm':
        trainer = train_Vanilla_utils(args, save_dir)
    trainer.setup()
    trainer.train()




