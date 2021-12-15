#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import copy
import torch
from glob import glob
#import datasets as ds
import os
from natsort import natsorted

from torchvision import datasets, transforms
from os import path, getcwd
import numpy as np

from models import MAX_SEQUENCE_LENGTH
from sampling import (
    mnist_iid,
    mnist_noniid,
    mnist_noniid_unequal,
    cifar_iid,
    cifar_noniid,
    ade_iid,
    ade_noniid,
    synthetic_segmentation_unequal
)
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch


from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
    MapTransform,
    Spacing,
    AsChannelFirst,
    ToTensor,
    #ConvertLabelBrats,
    RandRotate,
    NormalizeIntensity,
    Resize
)



def num_batches_per_epoch(num_datapoints: int, batch_size: int) -> int:
    """Determine the number of data batches depending on the dataset size

    Args:
        num_datapoints (int): Number of examples present in the dataset
        batch_size (int): Batch size requested by the algorithm

    Returns:
        int: Number of batches to use per epoch
    """
    num_batches, remaining_datapoints = divmod(num_datapoints, batch_size)

    if remaining_datapoints > 0:
        num_batches += 1

    return num_batches


def get_dataset(args, tokenizer=None, max_seq_len=MAX_SEQUENCE_LENGTH, custom_sampling=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.task == 'nlp':
        assert args.dataset == "ade", "Parsed dataset not implemented."
        [complete_dataset] = ds.load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split=["train"])
        # Rename column.
        complete_dataset = complete_dataset.rename_column("label", "labels")
        complete_dataset = complete_dataset.shuffle(seed=args.seed)
        # Split into train and test sets.
        split_examples = complete_dataset.train_test_split(test_size=args.test_frac)
        train_examples = split_examples["train"]
        test_examples = split_examples["test"]

        # Tokenize training set.
        train_dataset = train_examples.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )
        train_dataset.set_format(type="torch")

        # Tokenize test set.
        test_dataset = test_examples.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )
        test_dataset.set_format(type="torch")

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Ade_corpus
            user_groups = ade_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Ade_corpus
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = ade_noniid(train_dataset, args.num_users)

    elif args.task == 'cv':
        if args.dataset == 'brats':
            brats_nifti__dir_paths = natsorted(glob(args.ROOT_DATA+'/'"**/**/"))
            print(len(brats_nifti__dir_paths))
            print(f"loading BraTS data from {args.ROOT_DATA}/")
            imtrans = Compose(
                [   LoadImage(image_only=True),
                    Spacing(
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear"),
                        image_only = True,
                    ),
                    #ScaleIntensity(),
                    #NormalizeIntensity(nonzero=True, channel_wise=True),
                    AsChannelFirst(),
                    ToTensor(),
                    #AddChannel(),
                    #EnsureType(),
                    #Resized,
                    #RandFlip(prob=0.5, spatial_axis=0),
                    RandRotate(),
                    NormalizeIntensity(nonzero=True, channel_wise=True),
                    Resize(spatial_size=(112,112,72))
                ]
            )

            segtrans = Compose(
                [   LoadImage(image_only=True),
                    #AsChannelFirst(),
                    ToTensor(),
                    AddChannel(),
                    ConvertLabelBrats(keys=(1,2,4)),
                    #EnsureType(),
                    #Resized,
                    Resize(spatial_size=(112,112,72),mode='nearest')
                ]
            )
        
            #The whole dataset will be sliced in terms of the indexes of each of the partitions of the "centralized" dataset read in the next 3 lines
            train_ids = tuple(open('./data/partitions/brats_centralized_train.txt').read().split('\n'))
            val_ids = tuple(open('./data/partitions/brats_centralized_validation.txt').read().split('\n'))
            test_ids = tuple(open('./data/partitions/brats_centralized_test.txt').read().split('\n'))
            print(len(train_ids),len(val_ids),len(test_ids))
            train_ids_centers = []
            val_ids_centers   = []
            test_ids_centers  = []
            train_val_ids_centers = []
            for i in range(1,5): #four centers for BraTS
                train_file = './data/partitions/federated_brats/brats_federated_training_center_'+str(i)+'.txt'
                valid_file =  './data/partitions/federated_brats/brats_federated_validation_center_'+str(i)+'.txt'
                test_file =  './data/partitions/federated_brats/brats_federated_test_center_'+str(i)+'.txt'
                train_ids_centers.append(tuple(open(train_file).read().split('\n')))
                val_ids_centers.append(tuple(open(valid_file).read().split('\n')))
                train_val_ids_centers.append(train_ids_centers[-1]+val_ids_centers[-1])
                test_ids_centers.append(tuple(open(test_file).read().split('\n')))    

            train_volumes_paths, train_labels_paths = [],[]
            validation_volumes_paths, validation_labels_paths = [],[]
            test_volumes_paths, test_labels_paths = [],[]

            for item in brats_nifti__dir_paths:
                if item.split('/')[-2] in set(train_ids):
                    train_volumes_paths.append(item+"stacked.nii.gz")
                    train_labels_paths.append(item+"gt.nii.gz")
                if item.split('/')[-2] in set(val_ids):
                    validation_volumes_paths.append(item+"stacked.nii.gz")
                    validation_labels_paths.append(item+"gt.nii.gz")
                if item.split('/')[-2] in set(test_ids):
                    test_volumes_paths.append(item+"stacked.nii.gz")
                    test_labels_paths.append(item+"gt.nii.gz")


            train_dataset = ArrayDataset(train_volumes_paths, imtrans, train_labels_paths, segtrans)
            val_dataset   = ArrayDataset(validation_volumes_paths, imtrans, validation_labels_paths, segtrans)

            #This two are for using the validation fraction in the FL code
            train_val_volume_paths = train_volumes_paths+validation_volumes_paths
            train_val_label_paths = train_labels_paths+validation_labels_paths


            train_val_dataset = ArrayDataset(train_val_volume_paths, imtrans, train_val_label_paths, segtrans)
            test_dataset   = ArrayDataset(test_volumes_paths, imtrans, test_labels_paths, segtrans)

            user_groups = {i:[] for i in range(4)}
            user_groups_test = {i:[] for i in range(4)}

            train_val_id_centers = train_ids_centers + val_ids_centers

            #Generating mapping between train idxs and clients
            for i in range(len(train_val_volume_paths)):
                #print(i)
                cur_volume_id = train_val_volume_paths[i].split('/')[-2]
                for j in range(len(train_val_ids_centers)):
                    for k in range(len(train_val_ids_centers[j])):
                        if cur_volume_id == train_val_ids_centers[j][k]:
                            user_groups[j].append(i)
                            break

            #Generating mapping between test idxs and clients
            for i in range(len(test_volumes_paths)):
                #print(i)
                cur_volume_id = test_volumes_paths[i].split('/')[-2]
                for j in range(len(test_ids_centers)):
                    for k in range(len(test_ids_centers[j])):
                        if cur_volume_id == test_ids_centers[j][k]:
                            user_groups_test[j].append(i)
                            break

            return train_val_dataset, test_dataset, user_groups, user_groups_test


        if args.dataset == 'cifar':
            data_dir = './data/cifar/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=apply_transform)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)

            # sample training data amongst users
            if custom_sampling is not None:
                user_groups = custom_sampling(dataset=train_dataset, num_users=args.num_users)
                assert len(user_groups) == args.num_users, "Incorrect number of users generated."
                check_client_sampled_data = []
                for client_idx, client_samples in user_groups.items():
                    assert len(client_samples) == len(train_dataset)/args.num_users, "Incorrectly sampled client shard."
                    for record in client_samples:
                        check_client_sampled_data.append(record)
                assert len(set(check_client_sampled_data)) == len(train_dataset), "Client shards are not i.i.d"
                print("Congratulations! You've got it :)")
            else:
                # sample training data amongst users
                if args.iid:
                    # Sample IID user data from Mnist
                    user_groups = cifar_iid(train_dataset, args.num_users)
                else:
                    # Sample Non-IID user data from Mnist
                    if args.unequal:
                        # Chose uneuqal splits for every user
                        raise NotImplementedError()
                    else:
                        # Chose euqal splits for every user
                        user_groups = cifar_noniid(train_dataset, args.num_users)
        if args.dataset =='synthetic':
            print("loading synthetic data...")
            data_dir = '.data/synthetic/'
            train_imtrans = transforms.Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                RandSpatialCrop((96, 96), random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                EnsureType(),
            ])
            train_segtrans = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                RandSpatialCrop((96, 96), random_size=False),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                EnsureType(),
            ])
            val_imtrans = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()])
            val_segtrans = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()])

            #Reading the image paths
            print(args.ROOT_DATA + 'train/' + args.dataset +'/')
            train_dir = args.ROOT_DATA + '/' + args.dataset + '/train/' 
            val_dir   = args.ROOT_DATA + '/' + args.dataset + '/val/' 
            test_dir  = args.ROOT_DATA + '/' + args.dataset + '/test/' 
            imagepaths_train = sorted(glob(os.path.join(train_dir, "*img*.png")))
            segmpaths_train   = sorted(glob(os.path.join(train_dir, "*seg*.png")))
            print(len(imagepaths_train),len(segmpaths_train))
            imagepaths_val = sorted(glob(os.path.join(val_dir, "*img*.png")))
            segmpaths_val   = sorted(glob(os.path.join(val_dir, "*seg*.png")))
            print(len(imagepaths_val),len(segmpaths_val))
            imagepaths_test = sorted(glob(os.path.join(test_dir, "*img*.png")))
            segmpaths_test   = sorted(glob(os.path.join(test_dir, "*seg*.png")))
            print(len(imagepaths_test),len(segmpaths_test))
            train_dataset = ArrayDataset(imagepaths_train, train_imtrans, segmpaths_train, train_segtrans)#np.zeros((10,10,10))
            test_dataset  = ArrayDataset(imagepaths_test, val_imtrans, segmpaths_test, val_segtrans)


            print("Num users: " + str(args.num_users))
            
            user_groups   =  synthetic_segmentation_unequal(train_dataset, imagepaths_train, args.num_users)#{0:{0,1,2,3,4,5},1:{6,7,8,9,10}}#custom_sampling(dataset=train_dataset, num_users=args.num_users)


        elif args.dataset == 'mnist' or 'fmnist':
            if args.dataset == 'mnist':
                data_dir = './data/mnist/'
            else:
                data_dir = './data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose unequal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose equal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)
        else:
            raise NotImplementedError(
                f"""Unrecognized dataset {args.dataset}.
                Options are: `cifar`, `mnist`, `fmnist`.
                """
            )
    
    
    else:
        raise NotImplementedError(
            f"""Unrecognised task {args.task}.
            Options are: 'nlp', 'cv' and 'segmentation'.
            """
        )

    return train_dataset, test_dataset, user_groups


def average_weights(local_trained_weights):
    """Returns the average of the weights.
    """
    # Initialize copy model weights with the untrained model weights.
    avg_weights = copy.deepcopy(local_trained_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_trained_weights)):
            avg_weights[key] += local_trained_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_trained_weights))
    return avg_weights


def exp_details(args):
    """Print only function that displays experiment details.
    """
    print('\nExperimental details:')
    print(f'    Task      : {args.task}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


class ConvertLabelBrats(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WC (Whole tumor)*** we want WC because flair shows whole tumor
    and ET (Enhancing tumor).
   """

    def __call__(self, data):
        first_label_map = torch.tensor(np.array(data) == 1, dtype=torch.uint8)
        second_label_map = torch.tensor(np.array(data) == 2,dtype=torch.uint8)
        third_label_map = torch.tensor(np.array(data) == 4,dtype=torch.uint8)
        return torch.vstack((first_label_map,second_label_map,third_label_map))