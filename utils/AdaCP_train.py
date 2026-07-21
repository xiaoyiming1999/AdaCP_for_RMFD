#!/usr/bin/python
# -*- coding:utf-8 -*-
import copy
import os
import time
import warnings

import numpy as np
import torch
import logging
from torch import optim, nn
import torch.nn.functional as F
from models import resnet18, wrn_16_4, mobilenet_half, vgg11, rsnet18
from models import UA_CP_loss, _ECELoss, _BS_loss, Temperature_scaling, _ECE_EM_loss
from datasets import MCC5_data_split, BJUT_data_split, THU_data_split, HUST_data_split, HNU_data_split
from datasets import Mean_std_process
from utils.reliability_diagram import _reliability_diagram

class train_AdaCP_utils(object):
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
        self.ece_loss = _ECELoss(n_bins=args.num_bins)
        self.UA_CP_loss = UA_CP_loss(num_cls=args.num_classes)
        self.BS_loss = _BS_loss(num_cls=args.num_classes)
        self.ece_em_loss = _ECE_EM_loss(num_bins=args.num_bins, device=self.device)

        # initialize trade parameter for each bin
        self.bin_trade = torch.ones(size=(1, args.num_bins))
        self.bin_trade = args.initial_trade * self.bin_trade
        self.bin_trade = self.bin_trade.to(self.device)

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
                        soft_outputs = F.softmax(outputs, dim=1)

                        one_hot_labels = F.one_hot(labels, args.num_classes)
                        one_hot_labels = torch.gt(one_hot_labels, 0)
                        confidences_true = soft_outputs[one_hot_labels]
                        indices = torch.bucketize(confidences_true, self.bins)
                        flags = F.one_hot(indices - 1, args.num_bins)
                        sample_trade = torch.sum(flags * self.bin_trade, dim=1)

                        # calculate loss function
                        loss = self.UA_CP_loss(outputs, labels, sample_trade)

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

                # Calculate trade parameter
                if phase == 'val':

                    log_outputs = F.softmax(all_outputs, dim=1)
                    pred_confidences, pred_labels = torch.max(log_outputs, 1)
                    sorted_pred_confidences, _ = torch.sort(pred_confidences)
                    sorted_pred_confidences = torch.reshape(sorted_pred_confidences, (1, -1))
                    # ten bins
                    index_list = [i/args.num_bins for i in range(1, args.num_bins)]
                    index_list = torch.tensor(index_list)
                    index_list = index_list.to(self.device)
                    # bin boundaries
                    boundaries = torch.quantile(sorted_pred_confidences, index_list)
                    # bin start
                    start = torch.tensor([0])
                    start = start.to(self.device)
                    # bin end
                    end = torch.tensor([1])
                    end = end.to(self.device)
                    # bins
                    self.bins = torch.cat([start, boundaries, end])
                    # calculate gaps
                    indices = torch.bucketize(pred_confidences, self.bins)
                    bin_accuracies = torch.zeros(args.num_bins, dtype=torch.float64)
                    bin_confidences = torch.zeros(args.num_bins, dtype=torch.float64)
                    for b in range(args.num_bins):
                        selected = torch.where(indices == b + 1)[0]
                        if len(selected) > 0:
                            bin_acc_bool = all_labels[selected] == pred_labels[selected]
                            bin_accuracies[b] = torch.mean(torch.tensor([1.0 if value else 0.0 for value in bin_acc_bool]))
                            bin_confidences[b] = torch.mean(pred_confidences[selected])
                    gaps = bin_confidences - bin_accuracies
                    gaps = gaps.to(self.device)

                    for i in range(args.num_bins):

                        if self.bin_trade[0, i] >= 0:
                            self.bin_trade[0, i] = self.bin_trade[0, i] * torch.exp(args.lamda * gaps[i])
                            self.bin_trade[0, i] = torch.clamp(self.bin_trade[0, i], 0, 0.3)
                            if self.bin_trade[0, i] < args.threshold:
                                self.bin_trade[0, i] = -1 * args.threshold
                        else:
                            self.bin_trade[0, i] = self.bin_trade[0, i] * torch.exp(-1 * args.lamda * gaps[i])
                            self.bin_trade[0, i] = torch.clamp(self.bin_trade[0, i], -0.15, 0)
                            if torch.abs(self.bin_trade[0, i]) < args.threshold:
                                self.bin_trade[0, i] = args.threshold

                    # temperature scaling
                    if epoch == args.epoch - 1:

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












