from framework import Scaffold, FedAvg, FedRod, Fedem, Centralized
from preprocessing import dataPreprocessing, check_dataset
from numpy import std, mean
import numpy as np

#hide the warnings from torchio, because affine matrices are different for each sample
import warnings
warnings.filterwarnings("ignore")

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  additional_modalities= [], multi_label=False,
                  clients=[], size_crop=100, nested=True, train=True):

    print("Experiment using the ", datapath, "dataset")
    tmp_test = []
    tmp_valid = []

    #fetch the files paths, create the data loading/augmentation routines
    partitions_paths, partitions_paths_add_mod = dataPreprocessing(datapath, modality, clients, additional_modalities, nested, multi_label)

    if len(clients)<1:
        print("Must have at least one client")
        return None, None

    for i, conf in enumerate(networks_config):
        test_dicemetric = []
        valid_dicemetric = []
        for rep in range(num_repetitions):
            print(f"{networks_name[i]} iteration {rep+1}")
            print(conf)
            
            conf["partitions_paths"]=partitions_paths
            conf["partitions_paths_add_mod"]=partitions_paths_add_mod
                
            #add number to differentiate replicates
            if exp_name!=None:
                conf["suffix"]="_"+exp_name+"_"+str(rep)
            else:
                conf["suffix"]="_"+str(rep)

            conf["network_name"] = networks_name[i]

            if "scaff" in conf.keys() and conf["scaff"]:
                network = Scaffold(conf)
            elif "fedrod" in conf.keys() and conf["fedrod"]:
                network = FedRod(conf)
            elif 'weighting_scheme' in conf.keys():
                network = FedAvg(conf)
            elif "centralized" in conf.keys() and conf["centralized"]:
                network = Centralized(conf)
            else:
                print("missing argument for network type")

            if train:
                #train the network, each batch contain one ranodm slice for each subject
                network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'], early_stop_limit=conf['early_stop_limit'])

            # compute validation and test dice loss/score using full volume (instead of slice-wise) and the best possible model
            valid_dicemetric.append(network.full_volume_metric(dataset="valid", network="best", save_pred=True)[0])
            test_dicemetric.append(network.full_volume_metric(dataset="test", network="best", save_pred=True)[0])

        tmp_valid.append(valid_dicemetric)
        tmp_test.append(test_dicemetric)

    print("*** Summary for the experiment metrics ***")

    for k, (valid_metrics, test_metrics) in enumerate(zip(tmp_valid, tmp_test)):
        print(f"{networks_name[k]} valid avg dice: {mean(valid_metrics)} std: {std(valid_metrics)}")
        print(f"{networks_name[k]} test avg dice: {mean(test_metrics)} std: {std(test_metrics)}")

    return tmp_valid, tmp_test

if __name__ == '__main__':
    #path = 'astral_fedem_dti_purged/'
    #path = 'astral_fedem_dti/'
    #path = 'astral_fedem_dti_noempty/'
    #path = 'astral_fedem_v3/'
    #path = 'astral_fedem_dti_newlabels/'
    #path = 'astral_fedem_dti_noempty_newlabels/'
    #path = 'astral_fedem_4dir_1/'
    #path = 'astral_fedem_20dir/'
    path = 'astral_fedem_multiadc_newlabels/'
    #path = 'astral_fedem_ABC/'

    #experience_name = "astral_no_empty_mask"
    #experience_name = "no_empty_torchio_DLCE"
    #experience_name = "no_empty_tio_DLCE_newlabels" 
    #experience_name = "no_empty_DLCE_multiadc_transfo"
    #experience_name = "singlesite1_transfo"
    #experience_name = "singlesite2_no_transfo_blobloss"
    #experience_name = "allsite_transfo"
    experience_name = "multirep_siteB"
    
    modality="ADC"
    modality="20dir"

    clients=["center1", "center2", "center3"]
    #clients=["center3"]
    number_site=len(clients)

    default = {"g_epoch":60,
               "l_epoch":5,
               "g_lr":0.001,
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
               "loss_fun":"dicelossCE", #"blob_dicelossCE", #"dicelossCE", #diceloss_CE
               "hybrid_loss_weights":[1.4,0.6],
               #test time augmentation
               "use_test_augm":False,
               "test_augm_threshold":0.5, #at least half of the augmented img segmentation must agree to be labelled positive
               #adc subsampling augmentation/harmonization
               "no_deformation":True,
               "additional_modalities": [[],["4dir_1", "4dir_2"],[]] #list the extension of each additionnal modality you want to use for each site
               }

    #only used when using blob loss, labels are used to identify the blob
    default["multi_label"] = "blob" in default["loss_fun"]

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=5)

    #check that the additional_modalities argument has the good length
    assert len(clients)==len(default["additional_modalities"]), "additionnal modality and clients should have the same length"

    #TODO: if using blob loss, should check if the labeled masks exist?

    networks_config = []
    networks_name = []
    #storing the best parameters
    lr = 0.001694
    #weight_comb = [1.3,0.7]
    weight_comb = [1.4, 0.6]
    #for lr in np.linspace(1e-5, 1e-2, 5):
    #for lr in [0.0005985, 0.001694, 0.00994, 0.01164]:
    #for weight_comb in [[1, 1], [1.4,0.6], [1.6,0.4]]: #sum up to 2 to keep the same range as first experient with 1,1
    #for lr in [0.00994]:
    #for lr in [0.00994, 0.0116]:
    """
        tmp = default.copy()
        tmp.update({"centralized":True, "l_lr":lr, "hybrid_loss_weights":weight_comb})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}_lambdas{str(tmp['hybrid_loss_weights'][0])}_{str(tmp['hybrid_loss_weights'][1])}")
        #legacy network naming, no lambdas (valid for v1 to v4)
        #networks_name.append(f"{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}")
    """
    for g_lr, l_lr in zip([0.01, 0.001, 0.0001], [0.001]*3):
        tmp = default.copy()
        tmp.update({"weighting_scheme":"FEDAVG", "l_lr":l_lr, "g_lr":g_lr})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_FEDAVG_llr{l_lr}_glr{g_lr}_batch{tmp['batch_size']}_ge{tmp['g_epoch']}_le{tmp['l_epoch']}")

    fedrod = default.copy()
    fedrod.update({"fedrod":True})

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
                                                nested=False,
                                                train=True,
                                                additional_modalities=default["additional_modalities"],
                                                multi_label=default["multi_label"]) 
