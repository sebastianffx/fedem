#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import argparse

from os import path, getcwd

sys.path.insert(0, path.join(getcwd(), "..", ".."))
from models import NLP_MODEL_CONFIG


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='the fraction of data to go into test split.')
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=8,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--task', type=str, default='cv', help='task name')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    if args.task == "nlp":
        args.lr = NLP_MODEL_CONFIG[args.model]["lr"]
        args.optimizer = "adam"
        args.num_classes = 2
        args.dataset = "ade"
    return args

def print_train_stats(args, epoch, batch_idx, images, trainloader, loss, flow_type):

    if args.task == 'cv':
        if flow_type == 'baseline':
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()), end="")

    pass

def get_nb_args(task: str = 'cv'):
    common_parameters = {
        "num_users": 10,
        "test_frac": 0.1,
        "frac": 0.2,
        "epochs": 10,
        "local_ep": 10,
        "local_bs": 8,
        "momentum": 0.5,
        "norm": "batch_norm",
        "gpu": None,
        "iid": 1,
        "unequal": 0,
        "stopping_rounds": 10,
        "verbose": 1,
        "seed": 1,
    }
    if task == 'nlp':
        task_parameters = {
            "task": "nlp",
            "model": "tinybert",
            "dataset": "ade",
            "num_classes": 2,
            "optimizer": "adam",
            "lr": 2e-5,


        }
    elif task == 'cv':
        task_parameters = {
            "task": "cv",
            "model": "cnn",
            "optimizer": "sgd",
            "lr": 0.01,
            "dataset": "cifar",
            "num_classes": 10,
            "num_channels": 3,
            "kernel_num": 9,
            "kernel_sizes": "3,4,5",
            "num_filters": 32,
            "max_pool": "True",
        }
    else:
        raise NotImplementedError(
            f"Parsed task {task} not implemented."
        )
    args = argparse.Namespace(
        **common_parameters,
        **task_parameters,
    )
    return args