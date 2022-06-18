import os
import torch
import numpy as np
import nibabel as nb
import torchio as tio
from glob import glob
from monai.data import (
    ArrayDataset,
    GridPatchDataset,
    create_test_image_3d,
    PatchIter)
from monai.transforms import (
    Activations,
    AsChannelFirstD,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandRotate,
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
    Resized)

def dataPreprocessing(path, modality, number_site, additional_modalities, nested=True):

    partitions_paths = get_train_valid_test_partitions(path, modality, number_site, nested)

    if len(additional_modalities)>0:
        partitions_paths_add_mod = add_modalities(path, additional_modalities, num_centers=number_site)
        return partitions_paths, partitions_paths_add_mod
    else:
        return partitions_paths, [[[],[],[]] for i in range(number_site)]

def get_train_valid_test_partitions(path, modality, num_centers=4, nested=True):
    centers_partitions = [[] for i in range(num_centers)]
    if nested:
        for center_num in range(1,num_centers+1):
            center_paths_train  = sorted(glob(path+'center'+str(center_num)+'/train'+'/**/*'+modality+'*/*.nii'))
            center_paths_valid  = sorted(glob(path+'center'+str(center_num)+'/valid'+'/**/*'+modality+'*/*.nii'))
            center_paths_test   = sorted(glob(path+'center'+str(center_num)+'/test'+'/**/*'+modality+'*/*.nii'))
            center_lbl_paths_train  = sorted(glob(path+'center'+str(center_num)+'/train'+'/**/*OT*/*nii'))
            center_lbl_paths_valid  = sorted(glob(path+'center'+str(center_num)+'/valid'+'/**/*OT*/*nii'))
            center_lbl_paths_test   = sorted(glob(path+'center'+str(center_num)+'/test'+'/**/*OT*/*nii'))
            centers_partitions[center_num-1] = [[center_paths_train,center_paths_valid,center_paths_test],[center_lbl_paths_train,center_lbl_paths_valid,center_lbl_paths_test]]
            print("site", str(center_num), "data loader contains (train/valid/test) map", len(center_paths_train), len(center_paths_valid), len(center_paths_test))
            if (len(center_paths_train), len(center_paths_valid), len(center_paths_test)) != (len(center_lbl_paths_train), len(center_lbl_paths_valid), len(center_lbl_paths_test)):
                print("not same number of images and masks!")
    else:
        for center_num in range(1,num_centers+1):
            #simplest way to add the modalities/channel would be to iterate here over the additionnal modalities, so that center_paths_train contain a list of modalities for each subject
            center_paths_train  = sorted(glob(path+'center'+str(center_num)+'/train/'+f'*{modality.lower()}.nii*'))
            center_paths_valid  = sorted(glob(path+'center'+str(center_num)+'/valid/'+f'*{modality.lower()}.nii*'))
            center_paths_test   = sorted(glob(path+'center'+str(center_num)+'/test/' +f'*{modality.lower()}.nii*'))
            center_lbl_paths_train  = sorted(glob(path+'center'+str(center_num)+'/train/'+'*msk.nii*'))
            center_lbl_paths_valid  = sorted(glob(path+'center'+str(center_num)+'/valid/'+'*msk.nii*'))
            center_lbl_paths_test   = sorted(glob(path+'center'+str(center_num)+'/test/' +'*msk.nii*'))
            centers_partitions[center_num-1] = [[center_paths_train,center_paths_valid,center_paths_test],[center_lbl_paths_train,center_lbl_paths_valid,center_lbl_paths_test]]
            print("site", str(center_num), "data loader contains (train/valid/test) map", len(center_paths_train), len(center_paths_valid), len(center_paths_test))
            if (len(center_paths_train), len(center_paths_valid), len(center_paths_test)) != (len(center_lbl_paths_train), len(center_lbl_paths_valid), len(center_lbl_paths_test)):
                print("not same number of images and masks!")
    return centers_partitions

def add_modalities(path, modalities, num_centers=4):
    partitions_paths_add_mod = []
    for center_num in range(1,num_centers+1):
        center_paths_train = []
        center_paths_valid = []
        center_paths_test = []
        for modality in modalities:
            center_paths_train.append(sorted(glob(path+'center'+str(center_num)+'/train/'+f'*{modality}.nii*')))
            center_paths_valid.append(sorted(glob(path+'center'+str(center_num)+'/valid/'+f'*{modality}.nii*')))
            center_paths_test.append(sorted(glob(path+'center'+str(center_num)+'/test/' +f'*{modality}.nii*')))
        partitions_paths_add_mod.append([center_paths_train,center_paths_valid,center_paths_test])
    #centers_partitions_add_mod = [[train, valid, test] for each site]
    #train = [[modality 1 for all subject], [modality 2 for all subject], ...]
    return partitions_paths_add_mod

def center_dataloaders(partitions_paths_center, transfo, batch_size=2):
    #print("Images-Labels: " + str(len(partitions_paths_center)))
    #print(len(partitions_paths_center[0][0]))
    #print(len(partitions_paths_center[1][0]))    
    #print(len(partitions_paths_center[0][1]))
    #print(len(partitions_paths_center[1][1]))
    #print(len(partitions_paths_center[0][2]))
    #print(len(partitions_paths_center[1][2]))
    
    center_ds_train = ArrayDataset(partitions_paths_center[0][0], transfo['imtrans'],
                                   partitions_paths_center[1][0], transfo['segtrans'])
    center_train_loader = torch.utils.data.DataLoader(
    	center_ds_train, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    	)

    center_ds_valid = ArrayDataset(partitions_paths_center[0][1], transfo['imtrans'],
                                   partitions_paths_center[1][1], transfo['segtrans'])
    center_valid_loader = torch.utils.data.DataLoader(
    	center_ds_valid, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    	)

    center_ds_test = ArrayDataset(partitions_paths_center[0][2], transfo['imtrans_test'],
                                  partitions_paths_center[1][2], transfo['segtrans_test'])
    center_test_loader = torch.utils.data.DataLoader(
        center_ds_test, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    	)
    """
    center_ds_train = ArrayDataset(partitions_paths_center[0][0], transfo['debug'], partitions_paths_center[1][0], transfo['debug'])
    center_train_loader = torch.utils.data.DataLoader(
    	center_ds_train, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
    	)

    center_ds_valid = ArrayDataset(partitions_paths_center[0][1], transfo['debug'], partitions_paths_center[1][1], transfo['debug'])
    center_valid_loader = torch.utils.data.DataLoader(
    	center_ds_valid, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
    	)

    center_ds_test = ArrayDataset(partitions_paths_center[0][2], transfo['debug'], partitions_paths_center[1][2], transfo['debug'])
    center_test_loader = torch.utils.data.DataLoader(
        center_ds_test, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available()
    	)
    """
    #print("===========LEN LOADERS==========")
    #print(len(center_train_loader))
    #print(len(center_valid_loader))
    #print(len(center_test_loader))
    #print("=====================")

    return center_train_loader, center_valid_loader, center_test_loader

def create_transfo(modality, size_crop):
    #creating the dataloader for 10 ISLES volumes using the T_max and the CBF
    #For cbf we are windowing 1-1024
    #For tmax we'll window 0-60
    #For CBV we'll window 0-200
    max_intensity=30
    if modality =='CBF':
        max_intensity = 1200
    elif modality =='CBV':
        max_intensity = 200
    elif modality =='Tmax' or modality =='MTT':
        max_intensity = 30
    elif modality == 'ADC':
        max_intensity = 4000

    transfo = {}

    ### debug
    debug = Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
        ]
    )
    transfo['debug']=debug
    ###

    transfo['imtrans']= Compose(
        [   LoadImage(image_only=True),
            ScaleIntensity(minv=0.0, maxv=1), #indicate the range in which you want the output to be
            AddChannel(),
            #RandRotate90(prob=0.5, spatial_axes=[0, 1]), this rotation is too extreme, does not reflect natural situations
            RandRotate(range_x=0.4, prob=0.2, padding_mode="zeros", mode="nearest"), #bilinear interpolation go keep things smooths
            RandSpatialCrop((size_crop, size_crop, 1), random_size=False), #used to sample a single slice
            EnsureType(),
            #Resized
        ]
    )

    transfo['segtrans']= Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            #RandRotate90(prob=0.5, spatial_axes=[0, 1]),
            RandRotate(range_x=0.4, prob=0.2, padding_mode="zeros", mode="nearest"), #nearest to make sure the labels remain dicrete
            RandSpatialCrop((size_crop, size_crop, 1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )

    transfo['imtrans_neutral']= Compose(
        [   LoadImage(image_only=True),
            #RandScaleIntensity( factors=0.1, prob=0.5),
            ScaleIntensity(minv=0.0, maxv=1),
            AddChannel(),
            RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )

    transfo['segtrans_neutral']= Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )

    transfo['imtrans_test']= Compose(
        [   LoadImage(image_only=True),
            ScaleIntensity(minv=0.0, maxv=1),
            AddChannel(),
            #RandSpatialCrop((size_crop, size_crop,1), random_size=False), In test we would like to process ALL slices
            EnsureType(),
            #Resized
        ]
    )

    transfo['segtrans_test']= Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            #RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )
    return transfo

def generate_loaders(partitions_paths, batch_size, modality, size_crop=224):

    transfo = create_transfo(modality, size_crop)
    
    centers_data_loaders = []
    for i in range(len(partitions_paths)):#Adding all the centers data loaders
        centers_data_loaders.append(center_dataloaders(partitions_paths[i], transfo, batch_size))

    #merging the training/test/validation dataloaders for the centralized model
    partitions_train_imgs = [partitions_paths[i][0][0] for i in range(len(partitions_paths))]
    partitions_train_lbls = [partitions_paths[i][1][0] for i in range(len(partitions_paths))]

    all_ds_train = ArrayDataset([i for l in partitions_train_imgs for i in l], transfo['imtrans'],
                                [i for l in partitions_train_lbls for i in l], transfo['segtrans'])
    all_train_loader   = torch.utils.data.DataLoader(
        all_ds_train, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    )

    partitions_test_imgs = [partitions_paths[i][0][2] for i in range(len(partitions_paths))]
    partitions_test_lbls = [partitions_paths[i][1][2] for i in range(len(partitions_paths))]

    #For selecting the model and testing in the heldout partition we collect the valid and test data from ALL centers
    all_ds_test = ArrayDataset([i for l in partitions_test_imgs for i in l], transfo['imtrans'], #in the future, should use imtrans_test
                               [i for l in partitions_test_lbls for i in l], transfo['segtrans'])
    all_test_loader   = torch.utils.data.DataLoader(
        all_ds_test, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
    )

    partitions_valid_imgs = [partitions_paths[i][0][1] for i in range(len(partitions_paths))]
    partitions_valid_lbls = [partitions_paths[i][1][1] for i in range(len(partitions_paths))]

    all_ds_valid = ArrayDataset([i for l in partitions_valid_imgs for i in l], transfo['imtrans'], #in the future, should use imtrans_test
                                [i for l in partitions_valid_lbls for i in l], transfo['segtrans'])
    all_valid_loader   = torch.utils.data.DataLoader(
        all_ds_valid, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available()
    )

    return centers_data_loaders, all_test_loader, all_valid_loader, all_train_loader

def torchio_get_loader_partition(partition_paths_adc, partition_paths_labels, partition_paths_additional_modalities=[]):
    subjects_list = []

    #TODO: compute connected component on the label map, so that the blob loss can take advantage of it
    # this could be performed prior to the loading as well, be part of the dataset creation/organization

    for i in range(len(partition_paths_adc)):
        subjects_list.append(tio.Subject(
                                #by default, tio add a channel when loading 3D volume
                                adc=tio.ScalarImage(path=[partition_paths_adc[i]]+[add_mod[i] for add_mod in partition_paths_additional_modalities]),
                                label=tio.LabelMap(partition_paths_labels[i])
                                #option 1 : store each additionnal modality as a tio.ScalarImage and choose one per subject/epoch?
                                #**additional_modalities #unpack the additionnal modalities, one attribute per modality
                                #option 2 : tio.Image can store 4D images, still stack 3D image along the Channel axis
                                )
                            )
    return subjects_list

def torchio_create_transfo(clamp_min, clamp_max, padding, patch_size, no_deformation=False, forced_channel=-1):
    clamp = tio.Clamp(out_min=clamp_min, out_max=clamp_max)
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    spatial = tio.OneOf({
            tio.RandomAffine(): 0.6,
            tio.RandomElasticDeformation(): 0.2,        
            tio.RandomAffine(degrees=180): 0.2
            },
            p=0.75,
        )
    rotation = tio.RandomAffine(degrees=360)
    padding = tio.Pad(padding=padding) #padding is typicaly equals to half the size of the patch_size
    toCanon = tio.ToCanonical() #reorder the voxel and correct affine matrix to have RAS+ convention

    #due to tio.Lambda specifications, output must have input shape
    def sample_channel(input):
        """Take advantage of one-element slice to maintain dimensions
        """
        if input.shape[0]==1:
            return input
        elif forced_channel>0:
            if forced_channel>input.shape[0]:
                print(f"Trying to force a non-existing channel {forced_channel}, maximum is {input.shape[0]}")
            elif forced_channel==input.shape[0]:
                return input[forced_channel:None,:,:,:]
            else:
                return input[forced_channel:forced_channel+1,:,:,:]
        else:
            sampled_C = torch.randint(low=0,high=input.shape[0],size=(1,)).item()
            return input[sampled_C:sampled_C+1,:,:,:]

    #apply the channel selection to the adc map only, the label have only one channel
    select_channel = tio.Lambda(sample_channel, types_to_apply=tio.INTENSITY)    

    #normalization only, no spatial transformation or data augmentation
    transform_valid = tio.Compose([select_channel, clamp, toCanon, rescale])
    #just regular campling and normalization
    if no_deformation:
        #still require padding for the label based patches creation
        transform = tio.Compose([select_channel, clamp, toCanon, rescale, padding])
        return transform, transform_valid
    #more transformation: affine, rotation, elastic deformation and planar symmetry
    else:
        #randomFlip should probably be along axes="Right/Left" or "Anterior/Posterior" to take advantage of the symmetry of the brain
        # Superior or Inferior don't make sense for 2D net
        transform = tio.Compose([select_channel, clamp, toCanon, rescale, spatial, tio.RandomFlip(axes="R"), padding, rotation])
        return transform, transform_valid

def torchio_create_test_transfo():
    """ Transform for test-time augmentation, the transformation should have been seen during the training.
        Here, we consider the 90, 180 and 270 rotation to be covered by the randomAffine(degrees=360)
    """
    #lossless
    #default axes for RandomFlip (training) is 0
    Right_flip = tio.Flip(axes="R") #symmetry plane = right plane of the brain
    Superior_flip = tio.Flip(axes="S") #symmetry plane = superior plane of the brain --> index engineering when iterating over full volume

    #lossly, tio.Affine was found directly in the source code
    #scale < 1 means dezooming, 0.1 means zooming/dezooming of at most 10%
    #using scale = 0 to avoid any zooming and prevent interpolation prior to prediction averaging
    rotation90 = tio.Affine(scales=0, degrees=90, translation=0)
    rotation180 = tio.Affine(scales=0, degrees=180, translation=0) #this transformation is geometrically VERY close to Anterior Flipping due to the symmetry of the brain
    rotation270 = tio.Affine(scales=0, degrees=270, translation=0)

    gaussian = tio.RandomNoise()
    gamma = tio.RandomGamma()

    #non-invertible : blurring

    #no need to compose, they will be applied indepently

    #proof of work without the noise adding function
    #return [h_flip, v_flip, rotation90, rotation180, rotation270]

    #proof of work with reduced number of augmentation
    return [Right_flip, rotation90, rotation180]
    


def torchio_generate_loaders(partitions_paths, batch_size, clamp_min=0, clamp_max=4000, padding=(50,50,1), patch_size=(128,128,1),
                             max_queue_length=16, patches_per_volume=4, no_deformation=False,
                             partitions_paths_add_mod=[], forced_channel=-1):

    print("Using TORCHIO dataloader")

    transform, transform_valid = torchio_create_transfo(clamp_min=clamp_min, clamp_max=clamp_max,
                                                        padding=padding, patch_size=patch_size,
                                                        no_deformation=no_deformation, forced_channel=forced_channel)

    #patch has 0.7 prob of being centered on a label=1
    labels_probabilities = {0: 0.3, 1: 0.7}
    sampler_weighted_probs = tio.data.LabelSampler(
        patch_size=patch_size,
        label_name='label',
        label_probabilities=labels_probabilities,
    )

    centers_data_loaders = []
    #create 3 dataloader per site [train, validation, test]
    for i in range(len(partitions_paths)):
        # get the subject, create subject datatset, feed it to a queue for the path creation, then converted to a dataloader
        centers_data_loaders.append([torch.utils.data.DataLoader(tio.Queue(tio.SubjectsDataset(torchio_get_loader_partition(partitions_paths[i][0][0],
                                                                                                                            partitions_paths[i][1][0],
                                                                                                                            partitions_paths_add_mod[i][0]
                                                                                                                            ),
                                                                                               transform=transform
                                                                                               ),
                                                                          max_queue_length,
                                                                          patches_per_volume,
                                                                          sampler_weighted_probs
                                                                          ),
                                                                 batch_size=batch_size
                                                                 ),
                                     #validation and test don't need the patch sampler (hence no queue)
                                     torch.utils.data.DataLoader(tio.SubjectsDataset(torchio_get_loader_partition(partitions_paths[i][0][1],
                                                                                                                  partitions_paths[i][1][1],
                                                                                                                  partitions_paths_add_mod[i][1]
                                                                                                                  ),
                                                                                     transform=transform_valid
                                                                                     ),
                                                        
                                                                 batch_size=batch_size
                                                                 ),
                                     torch.utils.data.DataLoader(tio.SubjectsDataset(torchio_get_loader_partition(partitions_paths[i][0][2],
                                                                                                                  partitions_paths[i][1][2],
                                                                                                                  partitions_paths_add_mod[i][2]
                                                                                                                  ),
                                                                                     transform=transform_valid
                                                                                     ),
                                                        
                                                                 batch_size=batch_size
                                                                 ),
                                    ])

    #aggreate for centralized model and validation/testing across sites
    partitions_train_imgs = [partitions_paths[i][0][0] for i in range(len(partitions_paths))]
    partitions_train_lbls = [partitions_paths[i][1][0] for i in range(len(partitions_paths))]
    
    #get the number of additionnal modalities, taking from first site, training
    number_add_mod = len(partitions_paths_add_mod[0][0])
    partitions_train_add_mod = []
    for mod in range(number_add_mod):
        mod_holder = []
        for i in range(len(partitions_paths_add_mod)):
            mod_holder += partitions_paths_add_mod[i][0][mod] #i -> site, 0 -> train, train [[modality x for all subject], [modality y for all subjects], ...]
        partitions_train_add_mod.append(mod_holder)
    all_train_subjects = tio.SubjectsDataset(torchio_get_loader_partition([i for l in partitions_train_imgs for i in l],
                                                                          [i for l in partitions_train_lbls for i in l],
                                                                          partitions_train_add_mod),
                                                                          transform=transform)
    
    partitions_valid_imgs = [partitions_paths[i][0][1] for i in range(len(partitions_paths))]
    partitions_valid_lbls = [partitions_paths[i][1][1] for i in range(len(partitions_paths))]
    partitions_valid_add_mod = []
    for mod in range(number_add_mod):
        mod_holder = []
        for i in range(len(partitions_paths_add_mod)):
            mod_holder += partitions_paths_add_mod[i][1][mod]
        partitions_valid_add_mod.append(mod_holder)
    all_valid_subjects = tio.SubjectsDataset(torchio_get_loader_partition([i for l in partitions_valid_imgs for i in l],
                                                                          [i for l in partitions_valid_lbls for i in l],
                                                                          partitions_valid_add_mod),
                                                                          transform=transform_valid)

    partitions_test_imgs = [partitions_paths[i][0][2] for i in range(len(partitions_paths))]
    partitions_test_lbls = [partitions_paths[i][1][2] for i in range(len(partitions_paths))]
    partitions_test_add_mod = []
    for mod in range(number_add_mod):
        mod_holder = []
        for i in range(len(partitions_paths_add_mod)):
            mod_holder += partitions_paths_add_mod[i][2][mod]
        partitions_test_add_mod.append(mod_holder)
    all_test_subjects = tio.SubjectsDataset(torchio_get_loader_partition([i for l in partitions_test_imgs for i in l],
                                                                         [i for l in partitions_test_lbls for i in l],
                                                                         partitions_test_add_mod),
                                                                         transform=transform_valid)

    queue_train = tio.Queue(all_train_subjects, max_queue_length, patches_per_volume, sampler_weighted_probs)
    all_train_loader = torch.utils.data.DataLoader(queue_train, batch_size=batch_size)

    #validation and test don't need the patch sampler
    all_valid_loader = torch.utils.data.DataLoader(all_valid_subjects, batch_size=1)
    all_test_loader = torch.utils.data.DataLoader(all_test_subjects, batch_size=1)
    
    return centers_data_loaders, all_test_loader, all_valid_loader, all_train_loader 

def check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=10):
    bad_dim_files = []
    for i in range(1,number_site+1):
        
        bad_dim_files += check_volume("./"+path+"center"+str(i)+"/train/", dim, thres_neg_val=thres_neg_val, thres_lesion_vol=thres_lesion_vol)
        bad_dim_files += check_volume("./"+path+"center"+str(i)+"/valid/", dim, thres_neg_val=thres_neg_val, thres_lesion_vol=thres_lesion_vol)
        bad_dim_files += check_volume("./"+path+"center"+str(i)+"/test/", dim, thres_neg_val=thres_neg_val, thres_lesion_vol=thres_lesion_vol)

    if delete:
        #remove duplicates
        for f in list(set(bad_dim_files)):
            os.remove(f)
    else:
        print("TBD")
        # pad them with zeros? instead of deleting them?

def check_volume(path, dim, thres_neg_val=-1e-6, thres_lesion_vol=10):
    bad_files = []
    files_name=os.listdir(path)
    for f in files_name:
        tmp= nb.load(path+f).get_fdata()
        #check dimensions
        tmp_shape = tmp.shape
        if tmp_shape != dim:
            bad_files.append(path+f)
            print(path+f, tmp_shape)
        #check the negatives values
        if tmp.min()<thres_neg_val:
            print(path+f, "contains negative value")
        #check nans
        if np.isnan(tmp).sum() > 0:
            print(path+f, "contains NaN")

        #check segmentation labels
        if "msk." in f:
            if tmp.sum() < thres_lesion_vol:
                print(path+f, f"lesion volume is smaller than {thres_lesion_vol}")
                #remove the mask
                bad_files.append(path+f)
                #remove the associated map
                bad_files.append(path+f.replace("msk.", "adc."))
        #TODO: count the number of connected components?

    return bad_files