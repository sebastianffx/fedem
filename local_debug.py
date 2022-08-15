from preprocessing import check_dataset
from numpy import std, mean
import numpy as np
from experiment import runExperiment
import torch

#hide the warnings from torchio, because affine matrices are different for each sample
import warnings

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    
    path, clients, folder_struct = ['debug_dataset/'], ["center1", "center2"], "site_simple"
    #path, clients, folder_struct = '../../../../../downloads/dataset_ISLES22_rel1/', ["center1"], "OTHER"

    experience_name = "debug_only_adc" 
    modality="ADC"

    number_site=len(clients)

    #regular 2D Unet, used for all Antoine's experiments
    nn_params= {"spatial_dims":2,
                "in_channels":1,
                "out_channels":1,
                "channels":(16, 32, 64, 128),
                "strides":(2, 2, 2),
                "kernel_size":(3,3),
                "num_res_units":2,
                #"mu": 0.01, #{0.001, 0.01, 0.1, 0.5, 1}, must be optimized
                }

    default = {#federation parameters
               "g_epoch":30,
               "l_epoch":5,
               "g_lr":0.1,
               "l_lr":0.01,
               "K":len(clients),
               "clients":clients,
               #network parameters
               "nn_class":"unet", ## swin_unetr, unetr
               "nn_params":nn_params,
               "loss_fun":"dicelossCE", #"blob_dicelossCE", #"dicelossCE", #diceloss_CE
               "hybrid_loss_weights":[1.4,0.6],
               "suffix":"exp5",
               #training parameters
               "val_interval":2,
               "modality":modality.lower(),
               "space_cardinality":2, #or 3, depending if you have a 2D or 3D network
               "batch_size":2,
               'early_stop_limit':20,
               #preprocessing pipeline parameters
               "use_torchio":True,
               "clamp_min":0,
               "clamp_max":4000,
               "patch_size":(64,64,1),
               "padding":(32,32,0), #typically half the dimensions of the patch_size
               "max_queue_length":16,
               "patches_per_volume":4,
               "no_deformation":True, # Re used for the pre processing
               "additional_modalities": [[]], #[[],[],[]] #[[],["4dir_1", "4dir_2"],[]] #list the extension of each additionnal modality you want to use for each site
               "additional_labels":False,
               #test time augmentation
               "use_test_augm":True,
               "test_augm_threshold":0.5, #at least half of the augmented img segmentation must agree to be labelled positive,
               "use_isles22_metrics":True
               }

    if "isle" in path[0].lower():
        """
        # 3D Unet, used for all Antoine's experiments
        default["padding"]    = (32,32,16)
        default["patch_size"] = (64,64,32)
        default["space_cardinality"] = 3
        nn_params= {"img_size":default["patch_size"],
                    "in_channels":1,
                    "out_channels":1,
                    "feature_size":48}


        default["nn_params"] = nn_params
        default["nn_class"]  = "swin_unetr"
        """
        default["additional_modalities"] = [[]] # "adc" the adc maps will be used in addition to the dwi (default) to add "flair"
        default["no_deformation"] = 'isles'
    else:
        default["additional_modalities"] = [[] for i in range(number_site)]
        #default["additional_modalities"] = [[],["tra4_1", "tra4_2"]]

    #only used when using blob loss, labels are used to identify the blob
    default["multi_label"] = "blob" in default["loss_fun"]

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    #check_dataset(path, number_site, dim=(112,112,72), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=5)

    #check that the additional_modalities argument has the good length
    assert len(clients)==len(default["additional_modalities"]), "additionnal modality and clients should have the same length"

    networks_config = []
    networks_name = []
    #storing the best parameters
    lr = 0.001694
    weight_comb = [1,1]
    #for lr in np.linspace(1e-5, 1e-2, 5):
    #for lr in [0.0005985, 0.001694, 0.00994, 0.01164]:
    for weight_comb in [[1.4,0.6]]: #sum up to 2 to keep the same range as first experient with 1,1
        tmp = default.copy()
        #tmp.update({"centralized":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
        #tmp.update({"fedprox":True})
        tmp.update({"scaff":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_SCAFF_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}_lambdas{str(tmp['hybrid_loss_weights'][0])}_{str(tmp['hybrid_loss_weights'][1])}")
        #legacy network naming, no lambdas (valid for v1 to v4)
        #networks_name.append(f"{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}")

    fedrod = default.copy()
    fedrod.update({"fedrod":True})

    fedprox = default.copy()
    fedprox["nn_params"] = default["nn_params"].copy()
    fedprox["nn_params"]["mu"] = 0.1
    fedprox.update({"fedprox":True})

    scaff = default.copy()
    scaff.update({"scaff":True})

    fedavg = default.copy()
    fedavg.update({"weighting_scheme":"FEDAVG"})

    fedbeta = default.copy()
    fedbeta.update({"weighting_scheme":"BETA",
                    "beta_val":0.9})

    #networks_name = ["CENTRALIZED", "FEDROD", "SCAFFOLD", "FEDAVG", "FEDBETA"]
    #networks_config = [centralized, fedrod, scaff, fedavg, fedbeta]

     
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=clients,
                                                size_crop=144,
                                                folder_struct=folder_struct,
                                                train=True,
                                                additional_labels=default["additional_labels"],
                                                additional_modalities=default["additional_modalities"],
                                                multi_label=default["multi_label"])
