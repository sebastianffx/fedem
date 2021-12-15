#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.extmath import softmax
from tqdm import tqdm
from itertools import islice
from os import path, getcwd

sys.path.insert(0, path.join(getcwd(), "..", ".."))
from models import get_optimizer
import monai 
from monai.data import  decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)





def load_split_dataset(dataset, idxs, batch_size, task, shuffle=False):
    #print("Splitting the dataset with: ")
    #print(idxs)
    #print(dataset)
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
            #return torch.tensor(content.detach().clone()), torch.tensor(label.detach().clone())
            return content.detach().clone(), label.detach().clone()
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
        """Initialize the object with client datasets and parsed arguments.
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

        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_test = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

        (   
            self.trainloader,
            self.validloader,
            self.testloader
        ) = self.train_val_test(dataset=dataset, idxs=list(idxs))
        # Default criterion set to NLL loss function
        self.criterion =  monai.losses.DiceLoss(sigmoid=True)#nn.NLLLoss().to(self.device)

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
        #print("Computations made with the following legths: ")
        #print("Train: " + str(len(idxs_train)))
        #print("Valid: " + str(len(idxs_train)))
        #print("Test: " + str(len(idxs_test)))

        batch_size_val  = int(len(idxs_val)/10) if int(len(idxs_val)/10)>0 else 1
        batch_size_test = int(len(idxs_test)/10) if int(len(idxs_test)/10)>0 else 1
        
        #print("Ratio Train: " + str(self.args.local_bs))
        #print("Ratio Valid: " + str(batch_size_val))
        #print("Ratio Test: " + str(batch_size_test))
        
        trainloader = load_split_dataset(dataset=dataset, idxs=idxs_train, batch_size=self.args.local_bs, task=self.args.task, shuffle=False)
        validloader = load_split_dataset(dataset=dataset, idxs=idxs_val, batch_size=batch_size_val, task=self.args.task,)
        testloader  = load_split_dataset(dataset=dataset, idxs=idxs_test, batch_size=batch_size_test, task=self.args.task,)

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
            if self.args.dataset == 'synthetic':#Is a segmentation task
                #print("Infering segmentation masks")
                roi_size = (96, 96)
                sw_batch_size = 4
                losses = []
                for batch_idx, (images, labels) in enumerate(self.testloader):
                    val_images, val_labels = images.to(self.device), labels.to(self.device)
                    roi_size = (96, 96)
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    #val_outputs = torch.from_numpy(np.array([self.post_trans(i) for i in decollate_batch(val_outputs)]))
                    # compute metric for current iteration
                    #print("========")
                    #val_outputs = torch.from_numpy(np.array([self.post_trans(i) for i in decollate_batch(val_outputs)]))
                    
                    #print(val_outputs.shape,val_labels.shape)
                    self.dice_metric(y_pred=val_outputs, y=val_labels)
                    #print(len(val_outputs))
                    #print(val_outputs.shape)
                    #print(val_labels.shape)
                    for i in range(len(val_outputs)):
                        loss = self.criterion(val_outputs[i], labels[i])
                        losses.append(loss)
                metric = self.dice_metric.aggregate().item() 
                # reset the status for next validation round
                self.dice_metric.reset()
                return metric, torch.mean(torch.tensor(losses))


            else: #Is a classification task
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


def get_split_idxs(idxs, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1):
    assert (train_frac + val_frac + test_frac) == 1
    #print("The indices are: " +str(idxs))
    idxs_train = idxs[:int(train_frac * len(idxs))]
    idxs_val = idxs[int(train_frac * len(idxs)):int((train_frac + val_frac) * len(idxs))]
    idxs_test = idxs[int((train_frac + val_frac) * len(idxs)):]
    #print(idxs_train)
    #print(idxs_val)
    #print(idxs_test)
    return (idxs_train, idxs_val, idxs_test)



def test_inference(args, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    if args.dataset == 'synthetic':
        criterion =  monai.losses.DiceLoss(sigmoid=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    else:
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
    if args.task == 'cv':
        if args.dataset == 'synthetic':#Is a segmentation task
            #print("Infering segmentation masks")
            roi_size = (96, 96)
            sw_batch_size = 4
            losses,metrics = [],[]
            for test_image,test_label in test_dataset:
                test_image, test_label = test_image[np.newaxis,:].to(device), test_label[np.newaxis,:].to(device)
                log_probs = model(test_image)
                loss = criterion(log_probs, test_label)
                val_outputs = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
                dice_metric(y_pred=val_outputs, y=test_label)            
                metric = dice_metric.aggregate().item()
                losses.append(loss)
                metrics.append(metric)
            # reset the status for next validation round
            dice_metric.reset()
            return torch.mean(torch.tensor(metrics)), torch.mean(torch.tensor(losses))#Return AVG dice and loss on test_dataset

        else: #Its CV classification
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
            accuracy = correct/total
            return accuracy, loss

        
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: `nlp` and `cv`.
            """
        )