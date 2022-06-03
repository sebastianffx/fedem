from glob import glob
import torch
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
    RandSpatialCrop,
    ScaleIntensity,
    EnsureType,
    Resized)

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

def center_dataloaders(partitions_paths_center, transfo, batch_size=2):
    #print("Images-Labels: " + str(len(partitions_paths_center)))
    #print(len(partitions_paths_center[0][0]))
    #print(len(partitions_paths_center[1][0]))    
    #print(len(partitions_paths_center[0][1]))
    #print(len(partitions_paths_center[1][1]))
    #print(len(partitions_paths_center[0][2]))
    #print(len(partitions_paths_center[1][2]))
    
    center_ds_train = ArrayDataset(partitions_paths_center[0][0], transfo['imtrans'], partitions_paths_center[1][0], transfo['segtrans'])
    center_train_loader = torch.utils.data.DataLoader(
    	center_ds_train, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    	)

    center_ds_valid = ArrayDataset(partitions_paths_center[0][1], transfo['imtrans'], partitions_paths_center[1][1], transfo['segtrans'])
    center_valid_loader = torch.utils.data.DataLoader(
    	center_ds_valid, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available()
    	)

    center_ds_test = ArrayDataset(partitions_paths_center[0][2], transfo['imtrans_test'], partitions_paths_center[1][2], transfo['segtrans_test'])
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

def dataPreprocessing(path, modality, number_site, batch_size, size_crop=224, nested=True):
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

    imtrans = Compose(
        [   LoadImage(image_only=True),
            #RandScaleIntensity( factors=0.1, prob=0.5),
            ScaleIntensity(minv=0.0, maxv=max_intensity),
            AddChannel(),
            RandRotate90( prob=0.5, spatial_axes=[0, 1]),
            RandSpatialCrop((size_crop, size_crop, 1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )
    transfo['imtrans']=imtrans

    segtrans = Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            RandRotate90( prob=0.5, spatial_axes=[0, 1]),
            RandSpatialCrop((size_crop, size_crop, 1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )
    transfo['segtrans']=segtrans



    imtrans_neutral = Compose(
        [   LoadImage(image_only=True),
            #RandScaleIntensity( factors=0.1, prob=0.5),
            ScaleIntensity(minv=0.0, maxv=max_intensity),
            AddChannel(),
            RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )

    transfo['imtrans_neutral']=imtrans_neutral

    segtrans_neutral = Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )
    transfo['segtrans_neutral']=segtrans_neutral

    imtrans_test = Compose(
        [   LoadImage(image_only=True),
            ScaleIntensity(minv=0.0, maxv=max_intensity),
            AddChannel(),
            #RandSpatialCrop((size_crop, size_crop,1), random_size=False), In test we would like to process ALL slices
            EnsureType(),
            #Resized
        ]
    )
    transfo['imtrans_test']=imtrans_test

    segtrans_test = Compose(
        [   LoadImage(image_only=True),
            AddChannel(),
            #RandSpatialCrop((size_crop, size_crop,1), random_size=False),
            EnsureType(),
            #Resized
        ]
    )
    transfo['segtrans_test']=segtrans_test

    partitions_paths = get_train_valid_test_partitions(path, modality, number_site, nested)
    
    centers_data_loaders = []
    for i in range(len(partitions_paths)):#Adding all the centers data loaders
        centers_data_loaders.append(center_dataloaders(partitions_paths[i], transfo, batch_size))

    partitions_test_imgs = [partitions_paths[i][0][2] for i in range(len(partitions_paths))]
    partitions_test_lbls = [partitions_paths[i][1][2] for i in range(len(partitions_paths))]

    partitions_valid_imgs = [partitions_paths[i][0][1] for i in range(len(partitions_paths))]
    partitions_valid_lbls = [partitions_paths[i][1][1] for i in range(len(partitions_paths))]

    #For selecting the model and testing in the heldout partition we collect the valid and test data from ALL centers
    all_ds_test = ArrayDataset([i for l in partitions_test_imgs for i in l], transfo['imtrans'],
                               [i for l in partitions_test_lbls for i in l], transfo['segtrans'])
    """
    all_ds_test = ArrayDataset([i for l in partitions_test_imgs for i in l], transfo['debug'],
                               [i for l in partitions_test_lbls for i in l], transfo['debug'])
    """
    all_test_loader   = torch.utils.data.DataLoader(
        all_ds_test, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available()
    )


    all_ds_valid = ArrayDataset([i for l in partitions_valid_imgs for i in l], transfo['imtrans'],
                                [i for l in partitions_valid_lbls for i in l], transfo['segtrans'])
    """
    all_ds_valid = ArrayDataset([i for l in partitions_valid_imgs for i in l], transfo['debug'],
                                [i for l in partitions_valid_lbls for i in l], transfo['debug'])
    """
    all_valid_loader   = torch.utils.data.DataLoader(
        all_ds_valid, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    return partitions_paths, centers_data_loaders, all_test_loader, all_valid_loader