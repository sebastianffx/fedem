import matplotlib
import os
import time
import random
import monai
import numpy as np
from network import *
from framework import FedRod
from preprocessing import dataPreprocessing


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
local_epochs, global_epochs = 1, 60
#no sampling
K=len(clients)

local_lr, global_lr = 0.00031, 1 #np.sqrt(K)

_, centers_data_loaders, all_test_loader, _ = dataPreprocessing(isles_data_root, modality, number_site=4, batch_size =batch_size)

#move center 3 at the end of the dataloaders
#tmp = centers_data_loaders[2]
#centers_data_loaders[2]=centers_data_loaders[3]
#centers_data_loaders[3]=tmp

options = {'K': K, 'l_epoch': local_epochs, 'B': batch_size, 'g_epoch': global_epochs, 'clients': clients,
           'l_lr':local_lr, 'g_lr':global_lr, 'dataloader':centers_data_loaders, 'suffix': 'FedRod', 
           'scaffold_controls': False, 'fed_rod':True}


fed_rod = FedRod(options)

fed_rod.train_server(global_epoch=global_epochs, local_epoch=local_epochs, global_lr=global_lr, local_lr=local_lr)

fed_rod.nns

fed_rod.test(all_test_loader)



