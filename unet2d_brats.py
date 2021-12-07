#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,ArrayDataset, GridPatchDataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet

from torch.utils.tensorboard import SummaryWriter
import monai
from monai.transforms import (
    Activations,
    Activationsd,
    AddChannel,
    AsDiscrete,
    AsDiscreted,
    AsChannelFirst,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    Invertd,
    LoadImaged,
    LabelToMask,
    LoadImage,
    MapTransform,
    NormalizeIntensity,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandSpatialCrop,
    Resize,
    RandFlip,
    RandRotate,
    Spacing,
    Spacingd,
    ScaleIntensity, 
    SpatialCrop,
    ToTensor,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized
)
from monai.utils import set_determinism
from natsort import natsorted
import torch

from glob import glob
import os.path
import nibabel as nib
from nibabel import load, save, Nifti1Image


print_config()


# In[2]:


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# In[3]:


LIEU = 'sebastian_insel'

if LIEU=='sebastian_insel':
    root_exp  = '/home/sebastian/experiments/fedem/'
    root_data = '/media/sebastian/data/ASAP/BRATS_2019_ubelix/center_wise/'
    brats_nifti__dir_paths = natsorted(glob(root_data+"**/**/"))

if LIEU=='sebastian_laptop':
    root_exp  = '/Users/sebastianotalora/work/postdoc/federated_learning/fedem/'
    root_data = '/Users/sebastianotalora/work/postdoc/federated_learning/data/'
    brats_nifti__dir_paths = natsorted(glob(root_data+'brats/'"**/**/"))
len(brats_nifti__dir_paths)#Should be 259


# In[4]:


EXP_TYPE = 'CENTRALIZED' #FEDERATED
if EXP_TYPE == 'FEDERATED':
    print("To Implement")
if EXP_TYPE == 'CENTRALIZED':
    train_ids = tuple(open(root_exp+'data/partitions/brats_centralized_train.txt').read().split('\n'))
    val_ids = tuple(open(root_exp+'data/partitions/brats_centralized_validation.txt').read().split('\n'))
    test_ids = tuple(open(root_exp+'data/partitions/brats_centralized_test.txt').read().split('\n'))


# In[5]:


print(len(set(train_ids)), len(set(val_ids)),len(set(test_ids)))


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[7]:


set_determinism(seed=0)


# In[8]:


train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)


# In[9]:


brats_nifti_labels = []#natsorted(glob(root_brats_data+"**/"+'gt*nii*'))
brats_nifti_flair = []#natsorted(glob(root_brats_data+"**/"+'flair*nii*'))
brats_nifti_t1 = []#natsorted(glob(root_brats_data+"**/"+'t1*nii*'))
brats_nifti_t1ce = []# natsorted(glob(root_brats_data+"**/"+'t1*nii*'))
brats_nifti_t2 = [] #natsorted(glob(root_brats_data+"**/"+'t2*nii*'))
brats_niftis_stacked = []


# In[10]:


##This is how the whole stacked dataset was created

#for item in brats_nifti__dir_paths:
    #brats_nifti_labels.append(item+"gt.nii.gz")
    #brats_nifti_flair.append(item+"flair.nii.gz")
    #brats_nifti_t1.append(item+"t1.nii.gz")
    #brats_nifti_t1ce.append(item+"t1ce.nii.gz")
    #brats_nifti_t2.append(item+"t2.nii.gz")
    #brats_niftis_stacked.append(item+"stacked.nii.gz")
    
    #assert(os.path.isfile(brats_nifti_labels[-1])==os.path.isfile(brats_nifti_flair[-1])==os.path.isfile(brats_nifti_t1[-1])==os.path.isfile(brats_nifti_t1ce[-1])==os.path.isfile(brats_nifti_t2[-1]))
    #flair_sample      = nib.load(brats_nifti_flair[-1])
    #t1_sample         = nib.load(brats_nifti_t1[-1])
    #brats_nifti_t1ce_sample  = nib.load(brats_nifti_t1ce[-1])
    #brats_nifti_t2_sample    = nib.load(brats_nifti_t2[-1])
    #img_np = flair_sample.get_fdata()
    #stacked_4d_nifti = np.zeros((flair_sample.shape[0],flair_sample.shape[1],flair_sample.shape[2],4),dtype='float64')
    #stacked_4d_nifti[:,:,:,0] = flair_sample.get_fdata()
    #stacked_4d_nifti[:,:,:,1] = t1_sample.get_fdata()
    #stacked_4d_nifti[:,:,:,2] = brats_nifti_t1ce_sample.get_fdata()
    #stacked_4d_nifti[:,:,:,3] = brats_nifti_t2_sample.get_fdata()
    #print("Saving nifti "+str(count_saving) + " out of " + str(len(brats_nifti__dir_paths)))
    #out = Nifti1Image(stacked_4d_nifti, header=flair_sample.header, affine=flair_sample.affine)
    #save(out, item+'stacked.nii.gz')    
    #count_saving+=1


# In[11]:


count_saving = 0
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


# In[12]:


pds = ArrayDataset([1, 2, 3, 4], lambda x: x + 0.1)


# In[13]:


pds[0]


# In[14]:


print(len(train_volumes_paths), len(validation_volumes_paths), len(test_volumes_paths))


# In[15]:


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


# In[16]:


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


# In[17]:


train_ds    = ArrayDataset(train_volumes_paths, imtrans, train_labels_paths, segtrans)
validation_ds    = ArrayDataset(validation_volumes_paths, imtrans, validation_labels_paths, segtrans)
test_ds    = ArrayDataset(test_volumes_paths, imtrans, test_labels_paths, segtrans)


# In[18]:


train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(validation_ds, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)


# In[19]:


len(train_ds[0])


# In[20]:


print(train_ds[0][0].shape,train_ds[0][1].shape)#sample,C, H,W,D


# In[21]:


plt.imshow(train_ds[0][0][0,:,:,10])


# In[22]:


sample = 70
slice_num = 30
print(f"image shape: {train_ds[sample][0][0,:,:,slice_num].shape}")
plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(train_ds[sample][0][i,:,:,slice_num].detach().cpu(), cmap="gray")
plt.show()
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {train_ds[sample][1][0,:,:,slice_num].shape}")
plt.figure("label", (18, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(train_ds[sample][1][i,:, :, slice_num].detach().cpu()*255)
plt.show()


# In[23]:


device


# In[24]:


max_epochs = 300
val_interval = 1
VAL_AMP = False

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(logit_thresh=0.5)]
)


# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = False


# In[25]:


model.spatial_dims


# In[ ]:


best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

total_start = time.time()
for epoch in range(300):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:#train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data[0].to(device),
            batch_data[1].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // 1}"
            #f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir, "best_metric_model.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start


# In[30]:


print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


# In[31]:


plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice TC")
x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
y = metric_values_tc
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice WT")
x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
y = metric_values_wt
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice ET")
x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
y = metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.show()


# In[32]:


device


# In[33]:


torch.save(         model.state_dict(),
                    os.path.join(root_dir, "best_metric_model_brats.pth"),
                )


# In[38]:


model.load_state_dict(
    torch.load(os.path.join(root_dir, "best_metric_model_brats.pth"))
)

not_so_random_volume_id = 11

model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = validation_ds[not_so_random_volume_id][0].unsqueeze(0).to(device)
    roi_size = (128, 128, 64)
    sw_batch_size = 4
    val_output = inference(val_input)
    val_output = post_trans(val_output[0])
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(validation_ds[not_so_random_volume_id][0][i, :, :, 30].detach().cpu(), cmap="gray")
    plt.show()
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(validation_ds[not_so_random_volume_id][1][i, :, :, 30].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i, :, :, 30].detach().cpu())
    plt.show()


# In[47]:


#val_org_transforms = Compose(
#    [
#        LoadImaged(keys=["image", "label"]),
#        EnsureChannelFirstd(keys=["image"]),
#       ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
#        Orientationd(keys=["image"], axcodes="RAS"),
#        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#        EnsureTyped(keys=["image", "label"]),
#    ]
#)

#val_org_ds = DecathlonDataset(
#    root_dir=root_dir,
#    task="Task01_BrainTumour",
#    transform=val_org_transforms,
#    section="validation",
#    download=False,
#    num_workers=0,
#    cache_num=0,
#)
#val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=0)

post_transforms = Compose([
    #EnsureTyped(keys="pred"),
    #Invertd(
    #    keys="pred",
    #    transform=val_org_transforms,
    #    orig_keys="image",
    #    meta_keys="pred_meta_dict",
    #    orig_meta_keys="image_meta_dict",
    #    meta_key_postfix="meta_dict",
    #    nearest_interp=False,
    #    to_tensor=True,
    #),
    Activations(sigmoid=True),
    AsDiscrete(logit_thresh=0.5),
])


# In[107]:


model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model_brats.pth")))
model.eval()

with torch.no_grad():
    for test_data in test_ds:
        test_inputs =test_data[0].to(device)
        test_output = sliding_window_inference(
            inputs=test_inputs[np.newaxis,:,:,:,:],
            roi_size=(112,112,72),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        test_output = [post_transforms(i) for i in decollate_batch(test_output)]
        #val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        dice_metric(y_pred=test_output[0], y=test_data[1].to(device))
        dice_metric_batch(y_pred=test_output[0], y=test_data[1].to(device))

    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()

    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

print("Metric on original image spacing: ", metric)
print(f"metric_tc: {metric_tc:.4f}")
print(f"metric_wt: {metric_wt:.4f}")
print(f"metric_et: {metric_et:.4f}")

