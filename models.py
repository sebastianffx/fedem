#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os

import torch.nn.functional as nn_fnx

from torch import nn, optim
from os import path, getcwd, makedirs

sys.path.insert(0, path.join(getcwd(), "..", ".."))

# Define a common max sequence length that can be imported across modules.
MAX_SEQUENCE_LENGTH = 250

# Define a model name: model full name dict.
NLP_MODEL_CONFIG = {
    "tinybert": {
        "model_name": "prajjwal1/bert-tiny",
        "lr": 2e-5,
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "lr": 2e-5,
    },
    "roberta": {
        "model_name": "roberta-base",
        "lr": 2e-5,
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased",
        "lr": 2e-5,
    },
}

def get_model(args, img_size=None, nlp_model_dict=NLP_MODEL_CONFIG):
    """Return model (and tokenizer if task is NLP) based on the provided args.
    For task CV optional argument to return specific sized images.

    Args:
        args: Parsed input arguments.
        img_size: Return images resized to specified size.
            Defaults to None.
        nlp_model_dict: Dictionary containing model name(key) and initial_model_checkpoint name(value).
            Defaults to NLP_MODEL_NAME.

    Raises:
        NotImplementedError: When unimplemented configuration is parsed.

    Returns:
        if args.task is 'nlp' returns global_model and tokenizer.
        if args.task is 'cv' returns global_model.


    """
    if args.task == 'nlp':
        if args.model in nlp_model_dict.keys():
            model_name = nlp_model_dict[args.model]["model_name"]
            from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
            model_config = AutoConfig.from_pretrained(model_name, num_labels=args.num_classes)
            global_model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            raise NotImplementedError(
                f"""Unsupported model {args.model} passed.
                Options: {nlp_model_dict.keys()}.
                """
            )
        return global_model, tokenizer

    elif args.task == 'cv':
        # Build model.
        if args.model == 'cnn':
            # Convolutional neural network
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = VGG(vgg_name="VGG11")
            else:
                raise NotImplementedError(
                    f"""Unrecognised dataset {args.dataset}.
                    Options are: `mnist`, 'fmnist' and `cifar`.
                    """
                )

        elif args.model == 'mlp':
            # Multi-layer perceptron
            img_size = img_size
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)
        else:
            raise NotImplementedError(
                f"""Unrecognised model {args.model}.
                Options are: `cnn` and `mlp`.
                """
            )
        return global_model

    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )


def get_optimizer(args, model, weight_decay=1e-4):
    """Return optimizer using parsed parameters.

    Args:
        args: Parsed input arguments.
        model: Model who's parameters are used to set-up optimizer.
        weight_decay: Weight decay parameter for optimizer.

    Returns:
        optimizer: Model training optimizer.
    """
    # Set optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=weight_decay)
    return optimizer


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = nn_fnx.relu(nn_fnx.max_pool2d(self.conv1(x), 2))
        x = nn_fnx.relu(nn_fnx.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = nn_fnx.relu(self.fc1(x))
        x = nn_fnx.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn_fnx.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(nn_fnx.relu(self.conv1(x)))
        x = self.pool(nn_fnx.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn_fnx.relu(self.fc1(x))
        x = nn_fnx.relu(self.fc2(x))
        x = self.fc3(x)
        return nn_fnx.log_softmax(x, dim=1)


vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x =
        return nn_fnx.log_softmax(x, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = nn_fnx.dropout(x, .2)
        conv1_out = nn_fnx.relu(self.conv1(x_drop))
        conv2_out = nn_fnx.relu(self.conv2(conv1_out))
        conv3_out = nn_fnx.relu(self.conv3(conv2_out))
        conv3_out_drop = nn_fnx.dropout(conv3_out, .5)
        conv4_out = nn_fnx.relu(self.conv4(conv3_out_drop))
        conv5_out = nn_fnx.relu(self.conv5(conv4_out))
        conv6_out = nn_fnx.relu(self.conv6(conv5_out))
        conv6_out_drop = nn_fnx.dropout(conv6_out, .5)
        conv7_out = nn_fnx.relu(self.conv7(conv6_out_drop))
        conv8_out = nn_fnx.relu(self.conv8(conv7_out))

        class_out = nn_fnx.relu(self.class_conv(conv8_out))
        pool_out = nn_fnx.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out