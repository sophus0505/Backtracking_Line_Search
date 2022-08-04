# -*- coding: utf-8 -*-
"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf
Forked and inspired by https://github.com/kuangliu/pytorch-cifar
Train CIFAR10 with PyTorch
Learning rate finder using Backtracking line search with different batch sizes
and different starting learning rates
"""

# from log import backup, login, logout; backup(); login()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
import json
import pickle

from models import *

from utils import progress_bar, count_parameters, dataset, dataset_MNIST

from lr_backtrack import LRFinder, change_lr

all_batch_sizes = [12, 25, 50, 100, 200, 400, 800]
all_lr_starts = [100, 10, 1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_loss = loss_avg = 1e10  # best (smallest) training loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
patient = 0  # number of epochs waiting for improvement of best_acc or best_loss

cifar_dataset = 10  # CIFAR100 or 100
num_classes = cifar_dataset
momentum = 0.9

# Backtracking hyper-parameters
BT = 1  # using backtracking or not
lr_justified = True
alpha = 1e-4
beta = 0.5


num_iter = 20

save_paths = ['weights/', 'history', 'history/lr']
for save_path in save_paths:
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
# Model
net = ResNet18(num_classes, in_dim=1)
net_name = 'ResNet18'


print('Model:', net_name)
print('Number of parameters:', count_parameters(net),
      'Numbers of Layers:', len(list(net.parameters())))
net = net.to(device)

save_dir = save_path + net_name + '/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# weights_init = save_dir + net_name+'_CF'+str(num_classes)+'_init.t7'
# weights_best = save_dir + net_name+'_CF'+str(num_classes)+'_best.t7'
# history_path = save_dir + net_name+'_CF'+str(num_classes)+'_history.json'

weights_init = save_dir + net_name+'MNIST'+'_init.t7'
weights_best = save_dir + net_name+'MNIST'+'_best.t7'
history_path = save_dir + net_name+'MNIST'+'_history.json'

# cuda device
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint %s..' % weights_best)
    assert os.path.isfile(
        weights_best), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(weights_best)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
all_history = {}

lr_full = {}
alpha_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# training


def train(epoch):
    global best_loss, loss_avg, history, patient_train, patient, optimizer_name
    train_loss = correct = total = 0
    patient = min([patient_test, patient_train])

    if optimizer_name == "SGD":
        lr_finder_BT.backtrack(trainloader, alpha=alpha, beta=beta,
                               num_iter=num_iter, lr_justified=lr_justified)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        lr_finder_BT.optimizer.zero_grad()
        outputs = lr_finder_BT.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        lr_finder_BT.optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        loss_avg = train_loss/(batch_idx+1)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)| LR: %.7f'
                     % (loss_avg, acc, correct, total, lr_finder_BT.lr_current))

    history['acc_train'].append(acc)
    history['loss_train'].append(loss_avg)
    history['lr'].append(lr_finder_BT.lr_current)

    # stop if no improvement over time
    if loss_avg > best_loss or np.isnan(loss_avg):
        patient_train += 1
        print('Total training loss does not decrease in last %d epoch(s)' % (
            patient_train))
    else:
        patient_train = 0
        best_loss = loss_avg

# Testing


def test(epoch):
    global history, patient_train, patient_test, best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = lr_finder_BT.model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            loss_avg = test_loss/(batch_idx+1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (loss_avg, 100.*correct/total, correct, total))

    history['acc_valid'].append(acc)
    history['loss_valid'].append(loss_avg)

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc:
        patient_test = 0
        print('Best valid accuracy!')
        best_acc = acc
    else:
        patient_test += 1
        print('Total valid accuracy does not increase in last %d epoch(s)' % (
            patient_test))


batch_size = 400
criterion = nn.CrossEntropyLoss()

lr_start = 0.1

all_history = {}

# trainloader, testloader, num_batches = dataset(cifar_dataset, batch_size)
trainloader, testloader, num_batches = dataset_MNIST(batch_size)

optimizers = {"SGD": optim.SGD(net.parameters(), lr=lr_start),
              "MMT": optim.SGD(net.parameters(), lr=lr_start, momentum=momentum),
              "NAG": optim.SGD(net.parameters(), lr=lr_start, momentum=momentum, nesterov=True)}
# "MMT": optim.SGD(net.parameters(), lr=lr_start, momentum=momentum),
# "NAG": optim.SGD(net.parameters(), lr=lr_start, momentum=momentum, nesterov=True),
alpha = 1e-6

for optimizer_name in optimizers:
    patient_train = 0
    patient_test = 0
    patient = 0
    best_acc = 0
    best_loss = loss_avg = 1e10  # best (smallest) loss

    optimizer_BT = optimizers[optimizer_name]
    if optimizer_name == "SGD":
        lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device=device)

    if not args.resume:
        if os.path.isfile(weights_init):
            print('Loading initialized weights from %s' % weights_init)
            net.load_state_dict(torch.load(weights_init))
        else:
            print('Saving initialized weights to %s' % weights_init)
            torch.save(net.state_dict(), weights_init)

    history = {}
    history['lr'] = []
    history['acc_train'] = []
    history['acc_valid'] = []
    history['loss_train'] = []
    history['loss_valid'] = []

    # main loop for training (with early stopping)
    for epoch in range(0, 10):
        if patient < 50:  # early stopping criteria
            print(f'{epoch = }, {alpha = }, {optimizer_name = }')
            train(epoch)
            test(epoch)

            all_history[alpha] = history
            json.dump(history, open(history_path, 'w'), indent=4)
            json.dump(all_history, open(
                "history/test%d.json" % (num_classes), 'w'), indent=4)
            pickle.dump(all_history, open(
                "history/test%d.pickle" % (num_classes), 'wb'))
