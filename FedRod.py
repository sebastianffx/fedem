import matplotlib
import os
import time
import random
import monai
import numpy as np
from network import *
from framework import FedRod
from preprocessing import dataPreprocessing
from numpy import std, mean
import nibabel as nib
import itertools
import torch
from monai.metrics import DiceMetric

lieu = 'SCAN' #SCAN or LAPTOP  or SERVER
isles_data_root = ''
if lieu == 'SCAN':
    isles_data_root = '/str/data/ASAP/miccai22_data/isles/federated/'
if lieu == 'LAPTOP':
    isles_data_root = '/Users/sebastianotalora/work/postdoc/data/ISLES/federated/'

exp_root = '/Users/sebastianotalora/work/tmi/fedem'
modality = 'Tmax'
batch_size = 2

clients=["center1", "center2", "center3", "center4"]
#from SCAFFOLD manuscript, global_learning_rate should be = sqrt(#Samples sites)
local_epochs, global_epochs = 1, 100
#no sampling
K=len(clients)

local_lr, global_lr = 0.00031, 1 #np.sqrt(K)

#move center 3 at the end of the dataloaders
#tmp = centers_data_loaders[2]
#centers_data_loaders[2]=centers_data_loaders[3]
#centers_data_loaders[3]=tmp

dices_repetitions = []
valid_dicemetric = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(5):
    test_dicemetric = []
    partitions_paths, centers_data_loaders, all_test_loader, all_valid_loader = dataPreprocessing(isles_data_root, modality, number_site=4, batch_size =batch_size)        
    options = {'K': K, 'l_epoch': local_epochs, 'B': batch_size, 'g_epoch': global_epochs, 'clients': clients,
            'l_lr':local_lr, 'g_lr':global_lr, 'dataloader':centers_data_loaders, 'suffix': 'FedRod', 
            'scaffold_controls': False, 'fed_rod':True, 'val_interval': 2, 'modality': modality, 'valid_loader': all_valid_loader, 'partitions_paths': partitions_paths}

    fed_rod = FedRod(options)
    fed_rod.train_server(global_epoch=global_epochs, local_epoch=local_epochs, global_lr=global_lr, local_lr=local_lr)
        
    partitions_test_imgs = [partitions_paths[i][0][2] for i in range(len(partitions_paths))]
    partitions_test_lbls = [partitions_paths[i][1][2] for i in range(len(partitions_paths))]
    all_test_paths  = list(itertools.chain.from_iterable(partitions_test_imgs))
    all_test_labels = list(itertools.chain.from_iterable(partitions_test_lbls))
    

    print("Loading model weights: ")
    model_path = '/home/otarola/miccai22/fedem/'+modality+'_FEDROD_best_metric_model_segmentation2d_array.pth'
    print(model_path)
    checkpoint = torch.load(model_path)
    fed_rod.nn.load_state_dict(checkpoint)
    model = fed_rod.nn

    pred = []
    y = []
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    dice_metric.reset()
    if modality =='CBF':
        max_intensity = 1200
    if modality =='CBV':
        max_intensity = 200
    if modality =='Tmax' or modality =='MTT':
        max_intensity = 30

    for path_test_case, path_test_label in zip(all_test_paths,all_test_labels):            
        test_vol = nib.load(path_test_case)
        test_lbl = nib.load(path_test_label)

        test_vol_pxls = test_vol.get_fdata()
        test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
        test_lbl_pxls = test_lbl.get_fdata()
        test_lbl_pxls = np.array(test_lbl_pxls)

        test_vol_pxls = (test_vol_pxls - 0) / (max_intensity - 0)
        dices_volume =[]
        slices_predictions, slices_gt = [],[]

        for slice_selected in range(test_vol_pxls.shape[-1]):
            out_test = model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
            out_test = out_test.detach().cpu().numpy()
            pred = np.array(out_test[0,0,:,:]>0.9, dtype='uint8')
            cur_dice_metric = dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
        test_dicemetric.append(dice_metric.aggregate().item())
        # reset the status for next computation round
        dice_metric.reset()
    dices_repetitions.append(np.mean(test_dicemetric))
print(dices_repetitions)
print(f"FedRod test avg dice: {np.mean(dices_repetitions)} std: {np.std(dices_repetitions)}")


