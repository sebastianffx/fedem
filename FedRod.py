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
    fed_rod.global_test()
print(dices_repetitions)
print(f"FedRod test avg dice: {np.mean(dices_repetitions)} std: {np.std(dices_repetitions)}")


