#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import StandardScaler
import sklearn


# In[2]:


LOCATION = 'scan' #laptop
if LOCATION == 'scan':
    isles_data_root = '/str/data/ASAP/miccai22_data/isles/federated/'
    exp_root = '/home/otarola/miccai22/fedem/'

if LOCATION == 'laptop':
    isles_data_root = '/data/ASAP/miccai22_data/isles/federated/'


# In[3]:
for i in range(5):

    #Hyperparams cell
    modality = 'CBF'
    batch_size = 2
    num_epochs = 300
    learning_rate = 0.000932#lrs[0] #To comment in the loop
    weighting_scheme = 'BETA'
    beta_val=0.9


    # In[4]:


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


    # In[5]:


    partitions_paths = get_train_valid_test_partitions(modality, isles_data_root, 4)


    # In[6]:


    len(partitions_paths[0][0][2]),len(partitions_paths[0][1][2]) #idx_order: center,img_label,partition


    # In[7]:


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


    # In[8]:


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


    # In[9]:


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


    # In[10]:


    centers_data_loaders = []
    for i in range(len(partitions_paths)):#Adding all the centers data loaders
        centers_data_loaders.append(center_dataloaders(partitions_paths[i],batch_size))


    # In[11]:


    len(centers_data_loaders)


    # In[12]:


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


    # In[13]:


    trainloaders_lengths = [len(centers_data_loaders[i][0].dataset) for i in [0,1,3]] #We don't take the Siemens training case

    beta = 0.9

    weight_classes = [(1-beta)/(1-np.power(beta,length)) for length in trainloaders_lengths]
    print(trainloaders_lengths)
    print(weight_classes)


    # In[14]:


    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    global_model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            kernel_size = (3,3),
            #dropout = 0.2,
            num_res_units=2,
    ).to(device)

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=learning_rate)



    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(comment=modality+"_"+weighting_scheme+"_LR_"+str(learning_rate)+"_BATCH_"+str(batch_size))


    batch_data = next(iter(centers_data_loaders[0][0]))


    # In[15]:


    inputs, labels = batch_data[0][:,:,:,:,0].to(device),batch_data[1][:,:,:,:,0].to(device)
    #torch.swapaxes(batch_data[0][0], 1, -1).to(device), torch.swapaxes(batch_data[1][0], 1, -1).to(device).to(device)
    print(inputs.shape,labels.shape)


    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    #Testing that the model works in one iteration
    optimizer.zero_grad()
    outputs = global_model(inputs)
    loss    = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()


    # In[16]:


    global_model.train()
    epoch_loss = 0
    train_loss, train_dice = [], []


    # In[17]:


    def perform_one_local_epoch(train_loader, local_model, local_optimizer, loss_function, client_idx=0):
        batch_loss_client = []
        for batch_data in train_loader:
            inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
            local_model.zero_grad()        
            outputs = local_model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            local_optimizer.step()
            batch_loss_client.append(loss.item())
        avg_loss_client = copy.deepcopy(sum(batch_loss_client) / len(batch_loss_client))
        print("Loss for client: " +str(client_idx)+" :" +str(avg_loss_client))
        return avg_loss_client   


    # In[18]:


    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"local epoch {epoch + 1}/{num_epochs}")

        local_weights, local_losses = [], []
        global_model.train()

        global_weights = global_model.state_dict()
        modelc1, modelc2, modelc4 = copy.deepcopy(global_model), copy.deepcopy(global_model), copy.deepcopy(global_model)
        optimizerc1 = torch.optim.Adam(modelc1.parameters(), learning_rate)
        optimizerc2 = torch.optim.Adam(modelc2.parameters(), learning_rate)
        optimizerc4 = torch.optim.Adam(modelc4.parameters(), learning_rate)

        modelc1.train()
        modelc2.train()
        modelc4.train()

        print(f"local epoch for train_loader 1: {epoch + 1}/{num_epochs}")
        loss_c1 = perform_one_local_epoch(centers_data_loaders[0][0], modelc1, optimizerc1, loss_function, client_idx=1)
        local_losses.append(loss_c1)
        print("Loss C1: " + str(local_losses[-1]))

        print(f"local epoch for train_loader 2: {epoch + 1}/{num_epochs}")
        loss_c2 = perform_one_local_epoch(centers_data_loaders[1][0], modelc2, optimizerc2, loss_function, client_idx=2)
        local_losses.append(loss_c2)
        print("Loss C2: " + str(local_losses[-1]))

        #C3 is the Siemens data loader for which we have only one data point
        print(f"local epoch for train_loader 4: {epoch + 1}/{num_epochs}")
        loss_c4 = perform_one_local_epoch(centers_data_loaders[3][0], modelc4, optimizerc4, loss_function, client_idx=4)
        local_losses.append(loss_c4)
        print("Loss C4: " + str(local_losses[-1]))

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)


        print(f"train_loss: {train_loss[-1]:.4f}")
        writer.add_scalar("train_loss", loss_avg, epoch)

        #Agregating the weights with the selected weighting scheme
        if weighting_scheme =='FEDAVG':
            global_weights = average_weights([copy.deepcopy(modelc1.state_dict()),copy.deepcopy(modelc2.state_dict()),copy.deepcopy(modelc4.state_dict())])
        if weighting_scheme =='BETA':
            global_weights = average_weights_beta([copy.deepcopy(modelc1.state_dict()),copy.deepcopy(modelc2.state_dict()),copy.deepcopy(modelc4.state_dict())],trainloaders_lengths,beta_val)
        if weighting_scheme =='SOFTMAX':
            global_weights = average_weights_softmax([copy.deepcopy(modelc1.state_dict()),copy.deepcopy(modelc2.state_dict()),copy.deepcopy(modelc4.state_dict())],trainloaders_lengths)

        
        # Update global weights with the averaged model weights.
        global_model.load_state_dict(global_weights)

        if (epoch + 1) % val_interval == 0:
            global_model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in all_valid_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = global_model(val_images[:,:,:,:,0])
                    val_outputs = val_outputs>0.5 #This assumes one slice in the last dim
                    dice_metric(y_pred=val_outputs, y=val_labels[:,:,:,:,0])
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(global_model.state_dict(), modality+'_beta_'+str(beta)+'_'+weighting_scheme+'_best_metric_model_segmentation2d_array.pth')
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)


    # In[20]:


    checkpoint = torch.load('/home/otarola/miccai22/fedem/'+modality+'_beta_'+str(beta)+'_'+weighting_scheme+'_best_metric_model_segmentation2d_array.pth')
    global_model.load_state_dict(checkpoint)
    outputs = global_model(inputs)
    print(modality)

    count_volume = 0
    dice_metric.reset()
    metric_values_test = []
    for test_data in all_test_loader:
        count_volume = count_volume+1
        cur_image, cur_label = test_data
        cur_outputs = []
        cur_labels  = []
        labels   = torch.tensor(cur_label).to(device)
        for ct_slice in range(cur_image.shape[-1]):
            cur_ct_slice = torch.tensor(cur_image[:,:,:,:,ct_slice]).to(device)        
            label    = labels[:,:,:,:,ct_slice]
            outputs = global_model(cur_ct_slice)

            cur_outputs.append(outputs.cpu().detach().numpy()>0.5)
            cur_labels.append(label.cpu().detach().numpy()>0.5)
        #print(torch.tensor(cur_outputs[-1]).shape)
        #print(torch.tensor(cur_labels[-1]).shape)
            dice_metric(y_pred=torch.tensor(cur_outputs[-1]), y=torch.tensor(cur_labels[-1]))

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values_test.append(metric)
    print("AVG TEST DICE SCORE FOR LEARNING RATE "+str(learning_rate) + ": " + str(np.mean(metric_values_test)) + " - STD: " + str(np.std(metric_values_test)))
    print(metric_values_test)


# In[ ]:




