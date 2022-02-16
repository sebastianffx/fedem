#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import monai
import numpy as np
import nibabel as nib
from glob import glob
from matplotlib import pyplot as plt

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


from torch.utils.tensorboard import SummaryWriter


# In[6]:


#loading volume paths
LOCATION = 'scan' #laptop
if LOCATION == 'scan':
    isles_data_root = '/str/data/ASAP/miccai22_data/isles/centralized/'
    exp_root = '/home/otarola/miccai22/fedem/'
if LOCATION == 'laptop':
    isles_data_root = '/data/ASAP/miccai22_data/isles/centralized/'

cbf_paths_train  = sorted(glob(isles_data_root+'/train/'+'/**/*CBF*/*.nii'))
tmax_paths_train = sorted(glob(isles_data_root+'/train/'+'/**/*Tmax*/*.nii'))
lbl_paths_train = sorted(glob(isles_data_root+'/train/'+'/**/*OT*/*nii'))


cbf_paths_valid  = sorted(glob(isles_data_root+'/valid/'+'/**/*CBF*/*.nii'))
tmax_paths_valid = sorted(glob(isles_data_root+'/valid/'+'/**/*Tmax*/*.nii'))
lbl_paths_valid = sorted(glob(isles_data_root+'/valid/'+'/**/*OT*/*nii'))


cbf_paths_test  = sorted(glob(isles_data_root+'/test/'+'/**/*CBF*/*.nii'))
tmax_paths_test = sorted(glob(isles_data_root+'/test/'+'/**/*Tmax*/*.nii'))
lbl_paths_test = sorted(glob(isles_data_root+'/test/'+'/**/*OT*/*nii'))

print(len(cbf_paths_train), len(tmax_paths_train), len(lbl_paths_train))
print(len(cbf_paths_valid), len(tmax_paths_valid), len(lbl_paths_valid))
print(len(cbf_paths_test), len(tmax_paths_test), len(lbl_paths_test))


# In[7]:


#creating the dataloader for 10 ISLES volumes using the T_max and the CBF
#For cbf we are windowing 1-1024
#For tmax we'll window 0-60
imtrans = Compose(
    [   LoadImage(image_only=True),
        ScaleIntensity(minv=0.0, maxv=1200),
        AddChannel(),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)

segtrans = Compose(
    [   LoadImage(image_only=True),
        AddChannel(),
        RandSpatialCrop((224, 224,1), random_size=False),
        EnsureType(),
        #Resized
    ]
)


# In[8]:


imtrans_test = Compose(
    [   LoadImage(image_only=True),
        ScaleIntensity(minv=0.0, maxv=1200),
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


cbf_paths_train


# In[10]:


ds_train = ArrayDataset(cbf_paths_train, imtrans, lbl_paths_train, segtrans)
train_loader   = torch.utils.data.DataLoader(
    ds_train, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
)

ds_valid = ArrayDataset(cbf_paths_valid, imtrans, lbl_paths_valid, segtrans)
valid_loader   = torch.utils.data.DataLoader(
    ds_valid, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
)

ds_test = ArrayDataset(cbf_paths_test, imtrans_test, lbl_paths_test, segtrans_test)
test_loader   = torch.utils.data.DataLoader(
    ds_test, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
)

im, seg = first(train_loader)
print(im.shape, seg.shape)
print(np.histogram(seg[0,0,:,:,0]))
print(im.max())



dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


# In[14]:


post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# In[15]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[121]:


#lrs = [np.random.rand(1)*1e-1, np.random.rand(1)*1e-2, np.random.rand(1)*1e-2, np.random.rand(1)*1e-3, np.random.rand(1)*1e-3]

print("IS CUDA AVAILABLE? " + str(torch.cuda.is_available()))
# In[16]:
lrs = [0.08132,0.08132,0.08132,0.08132]

for learning_rate in lrs:

    model = monai.networks.nets.UNet(
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
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)


    # In[18]:


    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()


    # In[19]:


    batch_data = next(iter(train_loader))
    batch_data[0].shape


    # In[20]:


    torch.swapaxes(batch_data[0][0], 1, -1).shape


    # In[21]:


    batch_data = next(iter(train_loader))
    inputs, labels = torch.swapaxes(batch_data[0][0], 1, -1).to(device), torch.swapaxes(batch_data[1][0], 1, -1).to(device)
    print(inputs.shape,labels.shape)


    # In[22]:


    model.train()



    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs)
    print(labels)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()


    # In[24]:


    for epoch in range(300):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{300}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            #inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            #Swaping axes to have a batch of Batch_size, Channels, width and height
            inputs, labels = torch.swapaxes(batch_data[0][0], 1, -1).to(device), torch.swapaxes(batch_data[1][0], 1, -1).to(device).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_loader) // train_loader.batch_size
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), exp_root+"/models/metric_model_segmentation2d_array_ep_"+str(epoch)+".pth")
        writer.add_scalar("average loss", epoch_loss, epoch + 1)
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in valid_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (128, 128)
                    sw_batch_size = 1
                    val_outputs = model(val_images[:,:,:,:,0])
                    val_outputs = val_outputs>0.5 #This assumes one slice in the last dim
                    #val_outputs = [sliding_window_inference(val_images[:,:,:,:,i], roi_size, sw_batch_size, model) for i in range(val_images.shape[-1])]
                    #val_outputs = torch.stack(val_outputs,dim=4)
                    #val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels[:,:,:,:,0])
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), exp_root+"/models/best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
        


    
    checkpoint = torch.load(exp_root+'/models/best_metric_model_segmentation2d_array.pth')

    count_volume = 0
    dice_metric.reset()
    metric_values_test = []
    for test_data in test_loader:
        count_volume = count_volume+1
        cur_image, cur_label = test_data
        cur_outputs = []
        cur_labels  = []
        labels   = torch.tensor(cur_label).to(device)
        for ct_slice in range(cur_image.shape[-1]):
            cur_ct_slice = torch.tensor(cur_image[:,:,:,:,ct_slice]).to(device)        
            label    = labels[:,:,:,:,ct_slice]
            outputs = model(cur_ct_slice)

            cur_outputs.append(outputs.cpu().detach().numpy()>0.5)
            cur_labels.append(label.cpu().detach().numpy()>0.5)
            dice_metric(y_pred=torch.tensor(cur_outputs[-1]), y=torch.tensor(cur_labels[-1]))

            #cur_save_img_path = '/home/otalora/'+cur_checkpoint_path.split('/')[-1] + '.png'
            #pp = np.array(outputs[0,0,:,:].cpu().detach().numpy()>0.5,dtype='uint8')
        cur_outputs = np.array(cur_outputs).squeeze()#.reshape(1,1,256,256,cur_image.shape[-1])#.squeeze()
        cur_labels  = np.array(cur_labels).squeeze()

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        metric_values_test.append(metric)
    print("AVG TEST DICE SCORE FOR LEARNING RATE "+str(learning_rate) + ": " + str(np.mean(metric_values_test)) + " - STD: " + str(np.std(metric_values_test)))
