#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import copy
import time
import pickle
import torch
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os import path, getcwd, makedirs
from tensorboardX import SummaryWriter

sys.path.insert(0, path.join(getcwd(), "..", ".."))

from options import args_parser
from update import ClientShard, test_inference
from models import get_model
from utils import (
    get_dataset,
    average_weights,
    exp_details,
)


if __name__ == '__main__':
    start_time = time.time()
    # define paths
    path_project = path.abspath('..')
    logger = SummaryWriter('./logs')

    args = args_parser()
    args.gpu = 'cuda:0'
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda:0' if args.gpu else 'cpu'

    if args.task == 'nlp':
        # Load NLP model and tokenizer.
        global_model, tokenizer = get_model(args)
        # Load tokenized dataset and user groups.
        train_dataset, test_dataset, user_groups = get_dataset(args, tokenizer)
    elif args.task == 'cv':
        # Load dataset and user groups.
        train_dataset, test_dataset, user_groups = get_dataset(args)
        # Load CV model.
        global_model = get_model(args=args, img_size=train_dataset[0][0].shape)
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )
    ### Start of Federated learning. {{{
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    #print(global_model)
    # copy weights
    global_weights = global_model.state_dict()
    # Training
    train_loss, train_accuracy = [], []
    print_every = 1
    exp_details(args)
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        global_model.train()
        # Randomly sample a fraction of clients and retrieve their ids.
        user_frac = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), user_frac, replace=False)

        for hidden_client_idx, idx in enumerate(idxs_users):
            client_shard = ClientShard(args=args,
                                       client_idx=hidden_client_idx,
                                       dataset=train_dataset,
                                       idxs=user_groups[idx],
                                       logger=logger,
                                       device=device)
            updated_local_model, loss = client_shard.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(updated_local_model))
            local_losses.append(copy.deepcopy(loss))

        # Calculate averaged model weights from all the client trained models.
        global_weights = average_weights(local_weights)
        # Update global weights with the averaged model weights.
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        # Remove trained clients from heldout evaluation.
        heldout_clients = list(range(args.num_users))
        for train_client_idx in list(idxs_users):
            heldout_clients.remove(train_client_idx)
        ###
        for heldout_client_idx, idx in tqdm(enumerate(heldout_clients),
                                            desc='Evaluating: Hidden client num:',
                                            total=len(heldout_clients)):
            client_shard = ClientShard(args=args,
                                       client_idx=heldout_client_idx,
                                       dataset=train_dataset,
                                       idxs=user_groups[idx],
                                       logger=logger,
                                       device=device)
            acc, loss = client_shard.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        ###
        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args=args, model=global_model, test_dataset=test_dataset, device=device)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    ### }}} End of Federated learning.

    # Saving the objects train_loss and train_accuracy:
    makedirs('./save/objects/', exist_ok=True)
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
