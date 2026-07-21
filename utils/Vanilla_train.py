#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import warnings

import numpy as np
import torch
import logging
from torch import optim, nn

from models import resnet18, wrn_16_4, mobilenet_half, vgg11, rsnet18
from models import _ECELoss, _BS_loss, Temperature_scaling, Focal_loss, Confidence, _ECE_EM_loss, \
    LabelSmoothingCrossEntropy, LogitNormLoss
from datasets import MCC5_data_split, BJUT_data_split, THU_data_split, HUST_data_split, HNU_data_split
from datasets import Mean_std_process
from utils.reliability_diagram import _reliability_diagram


class train_Vanilla_utils(object):
    def __init__(self, args, save_dir):
        self.save_dir = save_dir
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # define model
        if args.model == 'Resnet18':
            self.model = resnet18(out_channel=args.num_classes)
        elif args.model == 'WideResnet':
            self.model = wrn_16_4(num_cls=args.num_classes)
        elif args.model == 'MobileNet':
            self.model = mobilenet_half(in_c=args.in_channel, num_cls=args.num_classes)
        elif args.model == 'DRSN':
            self.model = rsnet18(out_channel=args.num_classes)
        elif args.model == 'vgg11':
            self.model = vgg11(in_c=args.in_channel, num_cls=args.num_classes)
        else:
            raise Exception("model not implement")

        # Define the learning parameters
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr}]

        # Define the optimizer
        self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        # Invert the model
        self.model.to(self.device)

        # define the loss
        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss = Focal_loss(num_class=args.num_classes)
        self.confidence_loss = Confidence()
        # Equal width ECE loss
        self.ece_loss = _ECELoss(n_bins=args.num_bins)
        self.BS_loss = _BS_loss(num_cls=args.num_classes)
        # Equal Mass ECE loss
        self.ece_em_loss = _ECE_EM_loss(num_bins=args.num_bins, device=self.device)
        self.label_smooth = LabelSmoothingCrossEntropy(num_cls=args.num_classes)
        self.lognorm = LogitNormLoss(device=self.device)

        # initialize bins
        self.bins = torch.linspace(0.0, 1.0, args.num_bins + 1)
        self.bins = self.bins.to(self.device)

    def train(self):

        args = self.args

        # load dataset
        self.datasets = {}

        if args.data_name == 'MCC5':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = MCC5_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                noise_SNR=args.noise_SNR,
                classes=args.classes,
                op_conditions=args.op_conditions,
            ).data_split()

        elif args.data_name == 'BJUT':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = BJUT_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                noise_SNR=args.noise_SNR,
                classes=args.classes,
                op_conditions=args.op_conditions,
            ).data_split()

        elif args.data_name == 'THU':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = THU_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                noise_SNR=args.noise_SNR,
                classes=args.classes,
                op_conditions=args.op_conditions,
            ).data_split()

        elif args.data_name == 'HUST':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = HUST_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                noise_SNR=args.noise_SNR,
                classes=args.classes,
                op_conditions=args.op_conditions,
            ).data_split()

        elif args.data_name == 'HNU':
            self.datasets['train'], self.datasets['val'], self.datasets['test'] = HNU_data_split(
                data_length=args.data_length,
                num_train_samples=args.num_train_samples,
                num_val_samples=args.num_val_samples,
                num_test_samples=args.num_test_samples,
                noise_SNR=args.noise_SNR,
                classes=args.classes,
                op_conditions=args.op_conditions,
            ).data_split()

        else:
            raise Exception("dataset not implement")

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False) for x in ['train', 'val', 'test']}

        for epoch in range(0, args.epoch):
            logging.info('-' * 20 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 20)
            #  learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val', 'test']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                all_outputs = []
                all_labels = []
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # mean-std normalize
                    inputs = Mean_std_process(inputs)
                    inputs = inputs.to(torch.float32)

                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        outputs = self.model(inputs)

                        # calculate loss function
                        if args.method == 'Vanilla':
                            loss = self.criterion(outputs, labels)
                        elif args.method == 'focal_loss':
                            loss = self.focal_loss(outputs, labels)
                        elif args.method == 'confidence':
                            loss = self.criterion(outputs, labels) - args.initial_trade * self.confidence_loss(outputs)
                        elif args.method == 'label_smooth':
                            loss = self.label_smooth(outputs, labels)
                        elif args.method == 'LogNorm':
                            loss = self.lognorm(outputs, labels)

                        all_outputs.append(outputs)
                        all_labels.append(labels)

                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                all_outputs = torch.cat(all_outputs)
                all_labels = torch.cat(all_labels)
                epoch_ece = self.ece_em_loss(all_outputs, all_labels)
                epoch_ece = epoch_ece.item()
                epoch_NLL = self.criterion(all_outputs, all_labels)
                epoch_NLL = epoch_NLL.item()
                epoch_BS = self.BS_loss(all_outputs, all_labels)
                epoch_BS = epoch_BS.item()

                # temperature scaling
                if phase == 'val' and epoch == args.epoch - 1:

                    # define temperature layer
                    self.temperature = Temperature_scaling()
                    self.temperature = self.temperature.to(self.device)

                    # define optimizer for temperature layer
                    parameter_list = [{"params": self.temperature.parameters(), "lr": 0.001}]
                    optimizer = optim.LBFGS(parameter_list, lr=0.001, max_iter=1000)

                    # train temperature layer using validation dataset
                    def closure():
                        optimizer.zero_grad()
                        calibrated_outputs = self.temperature(all_outputs)
                        loss = self.criterion(calibrated_outputs, all_labels)
                        loss.backward()
                        return loss
                    optimizer.step(closure)

                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info(
                    'Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, {}-ECE: {:.4f}, {}-NLL: {:.4f}, {}-BS: {:.4f}, Cost {:.1f} ms'.format(
                        epoch, phase, epoch_loss, phase, epoch_acc, phase, epoch_ece, phase, epoch_NLL, phase, epoch_BS,
                        1000 * (time.time() - epoch_start)
                    ))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # temperature for test dataset
        after_temperature_ece_for_test = self.ece_em_loss(self.temperature(all_outputs), all_labels)
        after_temperature_NLL_for_test = self.criterion(self.temperature(all_outputs), all_labels)
        after_temperature_BS_for_test = self.BS_loss(self.temperature(all_outputs), all_labels)
        logging.info('-' * 20 + 'After temperature' + '-' * 20)
        logging.info('After temperature: test-ECE: {:.4f}, test-NLL: {:.4f}, test-BS: {:.4f}'
                     .format(after_temperature_ece_for_test.item(), after_temperature_NLL_for_test.item(),
                             after_temperature_BS_for_test.item()))

        # plot reliability diagram
        true_labels = all_labels.detach().cpu().numpy()
        true_labels = true_labels.reshape(-1)
        softmaxes = all_outputs.softmax(dim=-1)
        confidences, predictions = torch.max(softmaxes, 1)
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        _reliability_diagram(true_labels=true_labels, pred_labels=predictions,
                             confidences=confidences, num_bins=args.num_bins)

