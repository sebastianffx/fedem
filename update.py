#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.extmath import softmax
from tqdm import tqdm
from itertools import islice
from os import path, getcwd

sys.path.insert(0, path.join(getcwd(), "..", ".."))
from models import get_optimizer


def get_split_idxs(idxs, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1):
    assert (train_frac + val_frac + test_frac) == 1

    idxs_train = idxs[:int(train_frac * len(idxs))]
    idxs_val = idxs[int(train_frac * len(idxs)):int((train_frac + val_frac) * len(idxs))]
    idxs_test = idxs[int((train_frac + val_frac) * len(idxs)):]

    return (idxs_train, idxs_val, idxs_test)


def load_split_dataset(dataset, idxs, batch_size, task, shuffle=False):
    splitloader = DataLoader(
        DatasetSplit(dataset, idxs, task),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return splitloader


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, task):
        self.dataset = dataset
        self.task = task
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.task == 'cv':
            content, label = self.dataset[self.idxs[item]]
            return torch.tensor(content), torch.tensor(label)
        elif self.task == 'nlp':
            return self.dataset[self.idxs[item]]
        else:
            raise NotImplementedError(
                f"""Unrecognised task {self.task}.
                Options are: `nlp` and `cv`.
                """
            )



class ClientShard(object):
    """A class that performs model fit on a client dataset.
    """
    def __init__(self, args, client_idx, dataset, idxs, logger, device):
        """Initialize the object with clinet datasets and parsed arguments.
        Args:
            args:
            dataset:
            idxs:
            logger:
        """
        self.args = args
        self.client_idx = client_idx
        self.logger = logger
        self.device = device
        (
            self.trainloader,
            self.validloader,
            self.testloader
        ) = self.train_val_test(dataset=dataset, idxs=list(idxs))
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        (
            idxs_train,
            idxs_val,
            idxs_test
        ) = get_split_idxs(idxs)

        trainloader = load_split_dataset(dataset=dataset, idxs=idxs_train, batch_size=self.args.local_bs, task=self.args.task, shuffle=True)
        validloader = load_split_dataset(dataset=dataset, idxs=idxs_val, batch_size=int(len(idxs_val)/10), task=self.args.task,)
        testloader = load_split_dataset(dataset=dataset, idxs=idxs_test, batch_size=int(len(idxs_test)/10), task=self.args.task,)

        return (trainloader, validloader, testloader)


    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = get_optimizer(args=self.args, model=model)

        if self.args.task == 'nlp':
            for epoch in range(self.args.local_ep):
                batch_loss = []
                # Iterate through batches and perform model parameter estimation.
                for (batch_idx, batch) in enumerate(self.trainloader):
                    model.train()
                    inputs = {
                        input_name: input_values.to(self.device)
                        for input_name, input_values in batch.items()
                    }
                    loss, *_ = model(**inputs, return_dict=False)
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()
                    loss.backward()
                    optimizer.step()

                    # For batches index in the multiples of 50, print training loss.
                    if self.args.verbose and (batch_idx % 4 == 0):
                        print('\r| Global Round : {} | Hidden client num : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, self.client_idx, epoch, batch_idx,
                            len(self.trainloader), 100. * batch_idx / len(self.trainloader), loss.item()), end="")
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                # Average the batch loss and append to epoch loss list.
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        elif self.args.task == 'cv':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('\r| Global Round : {} | Hidden client num : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, self.client_idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()), end="")
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            raise NotImplementedError(
                f"""Unrecognised task {self.args.task}.
                Options are: `nlp` and `cv`.
                """
            )

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        loss, total, correct = 0.0, 0.0, 0.0

        if self.args.task == 'nlp':
            scaled_batch_size = self.args.local_bs
            if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
                # NOTE: Multiple GPU devices exposed, evaluate using multiple GPU's.
                scaled_batch_size *= torch.cuda.device_count()
                model = nn.DataParallel(model)

            model.to(self.device)
            model.eval()

            predict_iterator = self.testloader

            with torch.no_grad():
                for batch_index, batch in enumerate(predict_iterator):
                    inputs = {
                        input_name: input_values.to(self.device)
                        for input_name, input_values in batch.items()
                    }
                    batch_loss , pred_logits, *_ = model(**inputs, return_dict=False)
                    loss += batch_loss.item()
                    pred_logits, pred_labels = torch.max(pred_logits, 1)
                    pred_labels = pred_labels.view(-1)
                    batch_labels = inputs["labels"].detach().cpu().numpy()
                    correct += torch.sum(torch.eq(pred_labels, torch.tensor(batch_labels))).item()
                    total += len(batch_labels)

        elif self.args.task == 'cv':
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
        else:
            raise NotImplementedError(
                f"""Unrecognised task {self.args.task}.
                Options are: `nlp` and `cv`.
                """
            )
        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    if args.task == 'nlp':
        scaled_batch_size = 128
        if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            # NOTE: Multiple GPU devices exposed, evaluate using multiple GPU's.
            scaled_batch_size *= torch.cuda.device_count()
            model = nn.DataParallel(model)

        model.to(device)
        model.eval()

        predict_iterator = tqdm(testloader, desc="Batch")

        with torch.no_grad():
            for batch_index, batch in enumerate(predict_iterator):
                inputs = {
                    input_name: input_values.to(device)
                    for input_name, input_values in batch.items()
                }
                batch_loss, pred_logits, *_ = model(**inputs, return_dict=False)
                loss += batch_loss.item()
                pred_logits, pred_labels = torch.max(pred_logits, 1)
                pred_labels = pred_labels.view(-1)
                batch_labels = inputs["labels"]
                correct += torch.sum(torch.eq(pred_labels, torch.tensor(batch_labels))).item()
                total += len(batch_labels)
    elif args.task == 'cv':
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )

    accuracy = correct/total
    return accuracy, loss
