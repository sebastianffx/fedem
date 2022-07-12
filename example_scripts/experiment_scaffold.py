#!/usr/bin/env python
# coding: utf-8

import sys

from utils_fedavg import get_optimizer
from weighting_schemes import average_weights, average_weights_beta, average_weights_softmax

import torch
from torch import nn, optim

import monai
import numpy as np
import nibabel as nib
from glob import glob
from matplotlib import pyplot as plt
import copy
from scipy.spatial import distance_matrix
from monai.transforms import (
    Activations,
    AsChannelFirstD,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
    Resized
)

from monai.data import (
    ArrayDataset, GridPatchDataset, create_test_image_3d, PatchIter)
from monai.utils import first
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, decollate_batch
from natsort import natsorted
import umap

from torch.utils.tensorboard import SummaryWriter

LOCATION = 'scan' #laptop
if LOCATION == 'scan':
    isles_data_root = '/home/madtoinou/pdm/data/federated/synthetic/'
    exp_root = '/home/madtoinou/pdm/data/'
if LOCATION == 'laptop':
    isles_data_root = '/data/ASAP/miccai22_data/isles/federated/'

#Hyperparams cell
modality = 'Tmax'
batch_size = 2
num_epochs = 300
learning_rate = 0.000932#lrs[0] #To comment in the loop
weighting_scheme = 'FEDAVG'
beta_val=0.9

#creating the dataloader for 10 ISLES volumes using the T_max and the CBF
#For cbf we are windowing 1-1024
#For tmax we'll window 0-60
#For CBV we'll window 0-200
if modality =='CBF':
    max_intensity = 1200
if modality =='CBV':
    max_intensity = 200
if modality =='Tmax' or modality =='MTT':
    max_intensity = 30

#Dataloaders

#Data augmentation operations
imtrans = Compose(
    [   LoadImage(image_only=True),
        #RandScaleIntensity( factors=0.1, prob=0.5),
        ScaleIntensity(minv=0.0, maxv=max_intensity),
        AddChannel(),
        RandRotate90( prob=0.5, spatial_axes=[0, 1]),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)

segtrans = Compose(
    [   LoadImage(image_only=True),
        AddChannel(),
        RandRotate90( prob=0.5, spatial_axes=[0, 1]),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)


imtrans_neutral = Compose(
    [   LoadImage(image_only=True),
        #RandScaleIntensity( factors=0.1, prob=0.5),
        ScaleIntensity(minv=0.0, maxv=max_intensity),
        AddChannel(),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)

segtrans_neutral = Compose(
    [   LoadImage(image_only=True),
        AddChannel(),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)

imtrans_test = Compose(
    [   LoadImage(image_only=True),
        ScaleIntensity(minv=0.0, maxv=max_intensity),
        AddChannel(),
        #RandSpatialCrop((224, 224,1), random_size=False), In test we would like to process ALL slices
        EnsureType(),
        #Resized
    ]
)

segtrans_test = Compose(
    [   LoadImage(image_only=True),
        AddChannel(),
        #RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)

def get_train_valid_test_partitions(modality, isles_data_root, num_centers=4):
    centers_partitions = [[] for i in range(num_centers)]
    for center_num in range(1,num_centers+1):
        center_paths_train  = sorted(glob(isles_data_root+'center'+str(center_num)+'/train'+'/**/*'+modality+'*/*.nii'))
        center_paths_valid  = sorted(glob(isles_data_root+'center'+str(center_num)+'/valid'+'/**/*'+modality+'*/*.nii'))
        center_paths_test   = sorted(glob(isles_data_root+'center'+str(center_num)+'/test'+'/**/*'+modality+'*/*.nii'))
        center_lbl_paths_train  = sorted(glob(isles_data_root+'center'+str(center_num)+'/train'+'/**/*OT*/*nii'))
        center_lbl_paths_valid  = sorted(glob(isles_data_root+'center'+str(center_num)+'/valid'+'/**/*OT*/*nii'))
        center_lbl_paths_test  = sorted(glob(isles_data_root+'center'+str(center_num)+'/test'+'/**/*OT*/*nii'))
        centers_partitions[center_num-1] = [[center_paths_train,center_paths_valid,center_paths_test],[center_lbl_paths_train,center_lbl_paths_valid,center_lbl_paths_test]]
    return centers_partitions

def center_dataloaders(partitions_paths_center, batch_size=2):#
    center_ds_train = ArrayDataset(partitions_paths_center[0][0], imtrans, partitions_paths_center[1][0], segtrans)
    center_train_loader   = torch.utils.data.DataLoader(
        center_ds_train, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    )

    center_ds_valid = ArrayDataset(partitions_paths_center[0][1], imtrans, partitions_paths_center[1][1], segtrans)
    center_valid_loader   = torch.utils.data.DataLoader(
        center_ds_valid, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    )

    center_ds_test = ArrayDataset(partitions_paths_center[0][2], imtrans_test, partitions_paths_center[1][2], segtrans_test)
    center_test_loader   = torch.utils.data.DataLoader(
        center_ds_test, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    )
    return center_train_loader, center_valid_loader, center_test_loader

def creater_dataloaders(modality, path, number_site, batch_size):
    partitions_paths = get_train_valid_test_partitions(modality, path, number_site)

    centers_data_loaders = []
    for i in range(len(partitions_paths)):#Adding all the centers data loaders
        centers_data_loaders.append(center_dataloaders(partitions_paths[i],batch_size))
        
    return partitions_paths, centers_data_loaders

partitions_paths, centers_data_loaders = creater_dataloaders(modality, isles_data_root, 4, batch_size)

# re-organization for validation and test
partitions_test_imgs = [partitions_paths[i][0][2] for i in range(len(partitions_paths))]
partitions_test_lbls = [partitions_paths[i][1][2] for i in range(len(partitions_paths))]

partitions_valid_imgs = [partitions_paths[i][0][1] for i in range(len(partitions_paths))]
partitions_valid_lbls = [partitions_paths[i][1][1] for i in range(len(partitions_paths))]

#For selecting the model and testing in the heldout partition we collect the valid and test data from ALL centers
all_ds_test = ArrayDataset([i for l in partitions_test_imgs for i in l],
                            imtrans, [i for l in partitions_test_lbls for i in l],
                            segtrans)
all_test_loader   = torch.utils.data.DataLoader(
    all_ds_test, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
)


all_ds_valid = ArrayDataset([i for l in partitions_valid_imgs for i in l],
                            imtrans, [i for l in partitions_valid_lbls for i in l],
                            segtrans)
all_valid_loader   = torch.utils.data.DataLoader(
    all_ds_valid, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#optimizer
class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

#network present in each client
class SCAF_unet(monai.networks.nets.UNet):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, kernel_size, num_res_units, name, E, lr):
        #call parent constructor
        super(SCAF_unet, self).__init__(spatial_dims=spatial_dims,
                                        in_channels=in_channels,
                                        out_channels=out_channels, 
                                        channels=channels,
                                        strides=strides,
                                        kernel_size=kernel_size, 
                                        num_res_units=num_res_units)

        self.name = name
        #control variables for SCAFFOLD
        self.E = E
        self.lr = lr
        self.control = {}
        self.delta_control = {}
        self.delta_y = {}

class Scaffold:
    def __init__(self, options):
        self.C = options['C']
        self.E = options['E']
        self.B = options['B']
        self.K = options['K']
        self.r = options['r']

        #save all clients dataloader
        self.dataloaders = options['dataloader']
        
        #server model
        self.nn = SCAF_unet(spatial_dims=2,
                            in_channels=1,
                            out_channels=1,
                            channels=(16, 32, 64, 128),
                            strides=(2, 2, 2),
                            kernel_size = (3,3),
                            num_res_units=2,
                            name='server',
                            E=options['E'],
                            lr=options['lr']).to(device)
        
        for k, v in self.nn.named_parameters():
            self.nn.control[k] = torch.zeros_like(v.data)
            self.nn.delta_control[k] = torch.zeros_like(v.data)
            self.nn.delta_y[k] = torch.zeros_like(v.data)
        
        #create clients
        self.nns = []
        for i in range(self.K):
            temp = copy.deepcopy(self.nn)
            temp.name = options['clients'][i]
            temp.control = copy.deepcopy(self.nn.control)  # ci
            temp.delta_control = copy.deepcopy(self.nn.delta_control)  # ci
            temp.delta_y = copy.deepcopy(self.nn.delta_y)
            temp.E = options['E']
            self.nns.append(temp)

    def train_server(self, epoch, learning_rate):
        for t in range(self.r):
            print('*** round', t + 1, '***')
            
            #skiping center 2 as only 1 scan is available
            #index=[0,1,3]
            
            #center 2 in included to check if it overfits (expected)
            index=[0,1,2,3]
            
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index, epoch, learning_rate)
            # aggregation
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0.0
        for j in index:
            # normal
            s += self.nns[j].len
        # compute
        x = {}
        c = {}
        # init
        for k, v in self.nns[0].named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                x[k] += self.nns[j].delta_y[k] / len(index)  # averaging
                c[k] += self.nns[j].delta_control[k] / len(index)  # averaging

        # update x and c
        for k, v in self.nn.named_parameters():
            v.data += x[k].data  # lr=1
            self.nn.control[k].data += c[k].data * (len(index) / self.K)

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index, epoch, learning_rate):  # update nn
        for k in index:
            self.nns[k] = train(self.nns[k], self.nn, k, self.dataloaders[k][0], epoch, learning_rate)

    def global_test(self, aggreg_dataloader_test):
        model = self.nn
        model.eval()
        
        #test the global model on each individual dataloader
        for k, client in enumerate(self.nns):
            print("testing on", client.name, "dataloader")
            test(model, self.dataloaders[k][2])
        
        #test the global model on aggregated dataloaders
        print("testing on all the data")
        test(model, aggreg_dataloader_test)
            
        

def train(ann, server, k, dataloader_train, epoch, learning_rate):
    #train client to train mode
    ann.train()
    ann.len = len(dataloader_train)

    print("training center", k)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    loss = 0
    x = copy.deepcopy(ann)
    optimizer = ScaffoldOptimizer(ann.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for epoch in range(epoch):
        for batch_data in dataloader_train:
            inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
            y_pred = ann(inputs)
            loss = loss_function(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(server.control, ann.control) #performing SGD on the control variables
        print("epoch", epoch, ":", loss.item())
    # update c
    # c+ <- ci - c + 1/(E * lr) * (x-yi)
    # save ann
    temp = {}
    for k, v in ann.named_parameters():
        temp[k] = v.data.clone()
    for k, v in x.named_parameters():
        ann.control[k] = ann.control[k] - server.control[k] + (v.data - temp[k]) / (ann.E * ann.lr)
        ann.delta_y[k] = temp[k] - v.data
        ann.delta_control[k] = ann.control[k] - x.control[k]
    return ann

def test(ann, dataloader_test):
    ann.eval()
    pred = []
    y = []
    for test_data in dataloader_test:
        with torch.no_grad():
            test_img, test_label = test_data[0].to(device), test_data[1].to(device)
            
            test_pred = ann(test_img[:,:,:,:,0])
            #WHY?
            test_pred =  test_pred>0.5 #This assumes one slice in the last dim
            
            dice_metric(y_pred=test_pred, y=test_label[:,:,:,:,0])
            
    # aggregate the final mean dice result
    metric = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()
    print('dice:', metric)

if __name__ == '__main__':
    clients=["center1", "center2", "center3", "center4"]
    C, E, B, r = 0.8, num_epochs, batch_size, 5
    #no sampling
    K=len(clients)
    learning_rate = 0.05
    local_epoch=10
    
    _, centers_data_loaders = creater_dataloaders(modality, isles_data_root, 4, batch_size)
    
    options = {'K': K, 'C': C, 'E': E, 'B': B, 'r': r, 'clients': clients,
               'lr':learning_rate, 'dataloader':centers_data_loaders}

    scaffold = Scaffold(options)
    scaffold.train_server(local_epoch, learning_rate)
    scaffold.global_test(all_test_loader)