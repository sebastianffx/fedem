from experiment import runExperiment
from preprocessing import check_dataset

if __name__ == '__main__':
    #path = 'astral_fedem_dti/'
    #path = 'astral_fedem_20dir/'
    #path = 'astral_fedem_multiadc_newlabels/'
    path = 'astral_fedem_ABC/'
    #path = 'astral_fedem_ABC_harmonized/'
    #path = 'astral_fedem_ABC_propermask/'
    #path = 'astral_fedem_ABC_harmonized_propermask/'

    #experience_name = "astral_no_empty_mask"
    #experience_name = "no_empty_torchio_DLCE"
    #experience_name = "no_empty_tio_DLCE_newlabels" 
    #experience_name = "no_empty_DLCE_multiadc_transfo"
    #experience_name = "singlesite1_transfo"
    #experience_name = "all20dir_deformation"
    experience_name = "ABC_nodeformation_subsampling"
    #experience_name = "all20dir_nodeformation"
    #experience_name = "B_nodeformation"
    #experience_name = "allsite_transfo_v1"
    #experience_name = "ABC_propermask_nodeformation"
    #experience_name = "singlerep_siteB_harmonized_propermask"
    #experience_name = "multirep_siteB_propermask"
    
    modality="ADC"
    #modality="20dir"

    clients=["center1", "center2", "center3"]
    #clients=["center2"]
    number_site=len(clients)

    #regular 2D Unet, used for all Antoine's experiments
    nn_params= {"spatial_dims":2,
                "in_channels":1,
                "out_channels":1,
                "channels":(16, 32, 64, 128),
                "strides":(2, 2, 2),
                "kernel_size":(3,3),
                "num_res_units":2}

    default = {#federation parameters
               "g_epoch":60,
               "l_epoch":5,
               "g_lr":0.001,
               "l_lr":0.001,
               "K":len(clients),
               "clients":clients,
               #network parameters
               "nn_class":"unet",
               "nn_params":nn_params,
               "loss_fun":"dicelossCE", #"blob_dicelossCE", #"dicelossCE", #diceloss_CE
               "hybrid_loss_weights":[1.4,0.6],
               "suffix":"exp5",
               #training parameters
               "val_interval":2,
               "modality":modality.lower(),
               "space_cardinality":2, #or 3, depending if you have a 2D or 3D network
               "batch_size":8,
               'early_stop_limit':20,
               #preprocessing pipeline parameters
               "use_torchio":True,
               "clamp_min":0,
               "clamp_max":4000,
               "patch_size":(128,128,1),
               "padding":(64,64,0), #typically half the dimensions of the patch_size
               "max_queue_length":16,
               "patches_per_volume":4,
               "no_deformation":True,
               "additional_modalities": [[],["4dir_1", "4dir_2"],[]], #list the extension of each additionnal modality you want to use for each site
               "additional_labels":False,
               #test time augmentation
               "use_test_augm":False,
               "test_augm_threshold":0.5, #at least half of the augmented img segmentation must agree to be labelled positive
               "use_isles22_metrics":True #compute isles22 metrics during validation
               }

    #only used when using blob loss, labels are used to identify the blob
    default["multi_label"] = "blob" in default["loss_fun"]

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=5)

    #check that the additional_modalities argument has the good length
    assert len(clients)==len(default["additional_modalities"]), "additionnal modality and clients should have the same length"

    networks_config = []
    networks_name = []
    #storing the best parameters
    lr = 0.001694
    weight_comb = [1.4, 0.6]
    """ 
    for lr in [0.00994]:
        tmp = default.copy()
        tmp.update({"centralized":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}_lambdas{str(tmp['hybrid_loss_weights'][0])}_{str(tmp['hybrid_loss_weights'][1])}")
    """
    """
    for g_lr in [0.0001]:
        for l_lr in [0.01]:
            tmp = default.copy()
            #tmp.update({"scaff":True, "l_lr":l_lr, "g_lr":g_lr})
            #tmp.update({"weighting_scheme":"BETA", "l_lr":l_lr, "g_lr":g_lr, "beta_val":0.9})
            tmp.update({"weighting_scheme":"FEDAVG", "l_lr":l_lr, "g_lr":g_lr})
            #tmp.update({"scaff":True, "l_lr":l_lr, "g_lr":g_lr})
            #tmp.update({"fedrod":True, "l_lr":l_lr, "g_lr":g_lr})
            networks_config.append(tmp)
            #do not forget to update the name of the framework used for the experiment!
            networks_name.append(f"{experience_name}_FEDAVG_llr{l_lr}_glr{g_lr}_batch{tmp['batch_size']}_ge{tmp['g_epoch']}_le{tmp['l_epoch']}")
     
    """
    for g_lr in [0.001]:
        for l_lr in [0.01]:
            for mu in [0.001]: 
                tmp = default.copy()
                tmp.update({"fedprox":True, "l_lr":l_lr, "g_lr":g_lr})
                tmp["nn_params"] = default["nn_params"].copy()
                tmp["nn_params"]["mu"] = mu 
                networks_config.append(tmp)
                networks_name.append(f"{experience_name}_FEDPROX_mu{mu}_llr{l_lr}_glr{g_lr}_batch{tmp['batch_size']}_ge{tmp['g_epoch']}_le{tmp['l_epoch']}")
    
        
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=clients,
                                                size_crop=144,
                                                folder_struct="site_simple",
                                                train=True,
                                                additional_modalities=default["additional_modalities"],
                                                additional_labels=default["additional_labels"],
                                                multi_label=default["multi_label"],
                                                use_isles22_metrics=True) 
    print("metrics for site 1 test set alone")
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=["center1"],
                                                size_crop=144,
                                                folder_struct="site_simple",
                                                train=False,
                                                additional_modalities=default["additional_modalities"],
                                                additional_labels=default["additional_labels"],
                                                multi_label=default["multi_label"],
                                                use_isles22_metrics=True) 
 
    print("metrics for site 2 test set alone")
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=["center2"],
                                                size_crop=144,
                                                folder_struct="site_simple",
                                                train=False,
                                                additional_modalities=default["additional_modalities"],
                                                additional_labels=default["additional_labels"],
                                                multi_label=default["multi_label"],
                                                use_isles22_metrics=True) 
    print("metrics for site 3 test set alone")
    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=1,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=["center3"],
                                                size_crop=144,
                                                folder_struct="site_simple",
                                                train=False,
                                                additional_modalities=default["additional_modalities"],
                                                additional_labels=default["additional_labels"],
                                                multi_label=default["multi_label"],
                                                use_isles22_metrics=True) 
