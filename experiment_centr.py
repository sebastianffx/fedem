from framework import Scaffold, FedAvg, FedRod, Fedem, Centralized
from preprocessing import get_train_valid_test_partitions, check_dataset
from numpy import std, mean
import numpy as np
from datetime import datetime


#hide the warnings from torchio, because affine matrices are different for each sample
import warnings
warnings.filterwarnings("ignore")

def check_config(config):
    if "spatial_dims" in config["nn_params"].keys():
        assert config["nn_params"]["spatial_dims"] == config["space_cardinality"]
    #TODO: verify more parameters

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  additional_modalities= [], additional_labels=False, multi_label=False,
                  clients=[], size_crop=100, folder_struct="site_nested", train=True):

    print("Experiment using the ", datapath, "dataset")
    tmp_test = []
    tmp_valid = []
    tmp_external = []

    #fetch the files paths, create the data loading/augmentation routines
    centers_partitions, \
    partitions_paths_add_mod, partitions_paths_add_lbl, \
    external_test, external_test_add_mod = get_train_valid_test_partitions(path=datapath,
                                                                           modality=modality,
                                                                           clients=clients,
                                                                           folder_struct=folder_struct,
                                                                           multi_label=multi_label,
                                                                           additional_modalities=additional_modalities,
                                                                           additional_labels=additional_labels)

    if len(clients)<1:
        print("Must have at least one client")
        return None, None

    for i, conf in enumerate(networks_config):
        test_dicemetric = []
        valid_dicemetric = []
        external_dicemetric = []

        #verify that the config parameters are coherent
        check_config(conf)

        for rep in range(num_repetitions):
            print(f"{networks_name[i]} iteration {rep+1}")
            print(conf)
            
            conf["partitions_paths"]=centers_partitions
            conf["partitions_paths_add_mod"]=partitions_paths_add_mod
            conf["partitions_paths_add_lbl"]=partitions_paths_add_lbl
            external_test = []
            external_test_add_mod = []
            conf["external_test"]= external_test 
            conf["external_test_add_mod"]=external_test_add_mod

            #add number to differentiate replicates
            if exp_name!=None:
                conf["suffix"]="_"+exp_name+"_"+str(rep)
            else:
                conf["suffix"]="_"+str(rep)

            conf["network_name"] = networks_name[i]
            #add number to differentiate replicates
            if exp_name!=None:
                conf["suffix"]="_"+exp_name+"_"
            if "scaff" in conf.keys() and conf["scaff"]:
                conf['summary_writer_name']= (f"runs/scaffold_llr{conf['l_lr']}_glr{conf['g_lr']}_le{conf['l_epoch']}_ge{conf['g_epoch']}_{conf['K']}sites_{conf['network_name']}_{conf['suffix']}_{rep}")
                network = Scaffold(conf)
            elif "fedrod" in conf.keys() and conf["fedrod"]:
                conf['summary_writer_name']= (f"runs/fedrod_llr{conf['l_lr']}_glr{conf['g_lr']}_le{conf['l_epoch']}_ge{conf['g_epoch']}_{conf['K']}sites__{conf['network_name']}_{conf['suffix']}_{rep}")
                network = FedRod(conf)
            elif 'weighting_scheme' in conf.keys():
                conf['summary_writer_name']= (f"runs/{conf['weighting_scheme']}_llr{conf['l_lr']}_glr{conf['g_lr']}_le{conf['l_epoch']}_ge{conf['g_epoch']}_{conf['K']}sites_{conf['network_name']}_{conf['suffix']}_{rep}")
                network = FedAvg(conf)
            elif "centralized" in conf.keys() and conf["centralized"]:
                conf['summary_writer_name']= (f"runs/central_llr{conf['l_lr']}_glr{conf['g_lr']}_le{conf['l_epoch']}_ge{conf['g_epoch']}_{conf['K']}sites_{conf['network_name']}_{conf['suffix']}_{rep}")
                network = Centralized(conf)
            else:
                print("missing argument for network type")
            #updating the summary writer name 

            if train:
                #train the network, each batch contain one ranodm slice for each subject
                network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'], early_stop_limit=conf['early_stop_limit'])

            # compute validation and test dice loss/score using full volume (instead of slice-wise) and the best possible model
            valid_dicemetric.append(network.full_volume_metric(dataset="valid", network="best", save_pred=False))
            test_dicemetric.append(network.full_volume_metric(dataset="test", network="best", save_pred=True))
            if len(external_test)>0:
                external_dicemetric.append(network.full_volume_metric(dataset="external_test", network="best", save_pred=False))
            #network="best" is redundant, we are reloading the same network for validation, test and external validation

        tmp_valid.append(valid_dicemetric)
        tmp_test.append(test_dicemetric)
        if len(external_test)>0:
            tmp_external.append(external_dicemetric)
        else:
            tmp_external.append(None)


    

    print("*** Summary for the experiment metrics ***")
    #average over the repetition of the same network
    for k, (valid_metrics, test_metrics, external_metrics) in enumerate(zip(tmp_valid, tmp_test, tmp_external)):
        print(f"{networks_name[k]} valid avg dice: {mean([tmp[0] for tmp in valid_metrics])} ({[tmp[0] for tmp in valid_metrics]}, std: {[tmp[1] for tmp in valid_metrics]}) global std: {np.round(std(valid_metrics),4)}")
        print(f"{networks_name[k]} test avg dice: {mean([tmp[0] for tmp in test_metrics])} ({[tmp[0] for tmp in test_metrics]}, std: {[tmp[1] for tmp in test_metrics]}) global std: {np.round(std(test_metrics),4)}")
        if len(external_test)>0:
            print(f"{networks_name[k]} external avg dice: {mean([tmp[0] for tmp in external_metrics])} ({[tmp[0] for tmp in external_metrics]}, std: {[tmp[1] for tmp in external_metrics]}) global std: {np.round(std(external_metrics),4)}")
    return tmp_valid, tmp_test

if __name__ == '__main__':

    path = '../../../../../../../../../../../home/jonathan/Dropbox/Documents/CHUV/ASAP/Data/isles_centers/'
    #path = 'astral_fedem_ABC_harmonized_propermask/'
    modality="CBF"
    #modality="20dir"


    clients=["center1", "center2", "center4"]
    #clients=["center1"]
    #clients=["center1"]
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
               "g_epoch":300,
               "l_epoch":1,
               "g_lr":0.001,
               "l_lr":0.001,
               "K":len(clients),
               "clients":clients,
               #network parameters
               "nn_class":"unet",
               "nn_params":nn_params,
               "loss_fun":"dicelossCE", #"blob_dicelossCE", #"dicelossCE", #diceloss_CE
               "hybrid_loss_weights":[1.999,0.0001],
               "suffix":"BRAINLESS",
               #training parameters
               "val_interval":2,
               "modality":modality,
               "space_cardinality":2, #or 3, depending if you have a 2D or 3D network
               "batch_size":8,
               'early_stop_limit':20,
               #preprocessing pipeline parameters
               "use_torchio":True,
               "clamp_min":0,
               "clamp_max":4000,
               "patch_size":(128,128,1),
               "padding":(32,32,0), #typically half the dimensions of the patch_size
               "max_queue_length":16,
               "patches_per_volume":4,
               "no_deformation":True,
               "additional_modalities": [[],[],[]], #[[],["4dir_1", "4dir_2"],[]] #list the extension of each additionnal modality you want to use for each site
               #test time augmentation
               "use_test_augm":False,
               "test_augm_threshold":0.5, #at least half of the augmented img segmentation must agree to be labelled positive
               "use_isles22_metrics":False, #compute isles22 metrics during validation
               #"weighting_scheme":"BETA",
               #"beta_val":0.99
               }

    experience_name = f"ISLES_Brainless_paper_{modality}_CENTRALIZED"
    #experience_name = "singlerep_siteB_harmonized_propermask"
    

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    #check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=5)

    #check that the additional_modalities argument has the good length
    assert len(clients)==len(default["additional_modalities"]), "additionnal modality and clients should have the same length"

    #only used when using blob loss, labels are used to identify the blob
    default["multi_label"] = "blob" in default["loss_fun"]

    #TODO: if using blob loss, should check if the labeled masks exist?

    networks_config = []
    networks_name = []
    #storing the best parameters
    lr_ = 0.00132
    #weight_comb = [1.3,0.7]
    weight_comb = [1.9999, 0.00001]
    #for lr in np.linspace(1e-5, 1e-2, 5):
    #for lr in [0.0005985, 0.001694, 0.00994, 0.01164]:
    #for weight_comb in [[1, 1], [1.4,0.6], [1.6,0.4]]: #sum up to 2 to keep the same range as first experient with 1,1
    #for lr in [0.00994]:
         
    # for lr in [lr_]:
    #     tmp = default.copy()
    #     # Getting the current date and time
    #     dt = datetime.now()
    #     ts = datetime.timestamp(dt)
    #     # getting the timestamp
    #     tmp.update({"centralized":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
    #     networks_config.append(tmp)
    #     networks_name.append(f"{ts}_{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}_lambdas{str(tmp['hybrid_loss_weights'][0])}_{str(tmp['hybrid_loss_weights'][1])}")
    #     #legacy network naming, no lambdas (valid for v1 to v4)

    for lr in [lr_]:
        tmp = default.copy()
        # Getting the current date and time
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        # getting the timestamp
        tmp.update({"centralized":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
        networks_config.append(tmp)
        networks_name.append(f"{ts}_{experience_name}_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}_lambdas{str(tmp['hybrid_loss_weights'][0])}_{str(tmp['hybrid_loss_weights'][1])}")
        

    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=4,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name=experience_name,
                                                modality=modality,
                                                clients=clients,
                                                size_crop=144,
                                                folder_struct="site_nested",
                                                train=True,
                                                additional_modalities=default["additional_modalities"],
                                                multi_label=default["multi_label"]) 
