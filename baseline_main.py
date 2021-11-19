#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import torch

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from itertools import islice
from os import makedirs, path, getcwd


from utils import get_dataset, num_batches_per_epoch
from options import args_parser
from update import test_inference
from models import get_model, get_optimizer


if __name__ == '__main__':
    args = args_parser()

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    if args.task == 'nlp':
        ### Start of natural language classification model training. {{{
        global_model, tokenizer = get_model(args=args)
        # Load tokenized dataset.
        train_dataset, test_dataset, _ = get_dataset(args=args, tokenizer=tokenizer)
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        # Training
        # Set optimizer and criterion
        optimizer = get_optimizer(args=args, model=global_model)
        # Prepare training set using `torch DataLoader`.
        trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=False)
        # Calculate number of training steps per epoch.
        steps_per_epoch = num_batches_per_epoch(num_datapoints=train_dataset.num_rows, batch_size=args.local_bs)

        epoch_loss = []
        global_model.zero_grad()
        epoch_iterator = trange(0, int(args.epochs), desc="Epoch")
        for epoch in epoch_iterator:
            batch_loss = []
            # Generate batches for a epoch.
            step_iterator = tqdm(
                islice(trainloader, steps_per_epoch),
                total=steps_per_epoch,
                desc="Batch",
                position=0,
                leave=True,
            )
            # Iterate through batches and perform model parameter estimation.
            for (batch_idx, batch) in enumerate(step_iterator):
                global_model.train()
                inputs = {
                    input_name: input_values.to(device)
                    for input_name, input_values in batch.items()
                }
                loss, *_ = global_model(**inputs, return_dict=False)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                global_model.zero_grad()
                # For batches index in the multiples of 50, print training loss.
                if batch_idx % 10 == 0:
                    print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx, len(trainloader),
                        100. * batch_idx / len(trainloader), loss.item()), end="")
                batch_loss.append(loss.item())
            # Average the batch loss and append to epoch loss list.
            loss_avg = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(loss_avg)
        ### }}} End of natural language classification model training.

    elif args.task == 'cv':
        ### Start of computer vision model training. {{{
        # Load dataset
        train_dataset, test_dataset, _ = get_dataset(args=args)
        # Get model
        global_model = get_model(args=args, img_size=train_dataset[0][0].shape)
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        # Training
        # Set optimizer and criterion
        optimizer = get_optimizer(args=args, model=global_model)
        criterion = torch.nn.NLLLoss().to(device)
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()), end="")
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(loss_avg)
        ### }}} End of computer vision model training.
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )
    # Plot loss
    makedirs('../save/', exist_ok=True)
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/{}_nn_{}_{}_{}.png'.format(args.task, args.dataset, args.model, args.epochs))

    # Testing
    test_acc, test_loss = test_inference(args=args, model=global_model, test_dataset=test_dataset, device=device)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
