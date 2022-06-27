from framework import Scaffold, FedAvg, FedRod, Fedem, Centralized
from preprocessing import dataPreprocessing, check_dataset
from numpy import std, mean
import numpy as np
from experiment import runExperiment

#hide the warnings from torchio, because affine matrices are different for each sample
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    path = 'astral_fedem_ABC/'

    experience_name = "fedavg_test"
    modality="ADC"

    clients=["center1", "center2", "center3"]
    number_site=len(clients)

    default = {"g_epoch":60,
               "l_epoch":5,
               "g_lr":0.01,
               "l_lr":0.001,
               "K":len(clients),
               "clients":clients,
               "suffix":"exp5",
               "val_interval":2,
               "modality":modality.lower(),
               "batch_size":8,
               'early_stop_limit':20,
               #all the parameters required to use the "new" torchio dataloader, a lot more of data augmentation
               "use_torchio":True,
               "clamp_min":0,
               "clamp_max":4000,
               "patch_size":(128,128,1),
               "padding":(64,64,0), #typically half the dimensions of the patch_size
               "max_queue_length":16,
               "patches_per_volume":4,
               "loss_fun":"dicelossCE", #diceloss_CE
               "hybrid_loss_weights":[1.4,0.6],
               #test time augmentation
               "use_test_augm":False,
               "test_augm_threshold":0.5, #at least half of the augmented img segmentation must agree to be labelled positive
               #adc subsampling augmentation/harmonization
               "no_deformation":False,
               "additional_modalities":[] #list the extension of each additionnal modality you want to use
               }

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=5)
    networks_config=[]
    networks_name=[]
    for g_lr, l_lr in zip([0.01, 0.001, 0.0001], [0.001]*3):
        tmp = default.copy()
        tmp.update({"weighting_scheme":"FEDAVG", "l_lr":l_lr, "g_lr":g_lr})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_FEDAVG_llr{l_lr}_glr{g_lr}_batch{tmp['batch_size']}_ge{tmp['g_epoch']}_le{tmp['l_epoch']}")
     
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                number_site=number_site,
                                                size_crop=144,
                                                nested=False,
                                                train=True,
                                                additional_modalities=[]) #default["additional_modalities"])
