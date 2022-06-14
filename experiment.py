from framework import Scaffold, FedAvg, FedRod, Fedem, Centralized
from preprocessing import dataPreprocessing, check_dataset
from numpy import std, mean
import numpy as np

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  number_site=3, size_crop=100, nested=True):

    print("Experiment using the ", datapath, "dataset")
    tmp_test = []
    tmp_valid = []

    #fetch the files paths, create the data loading/augmentation routines
    partitions_paths, transfo = dataPreprocessing(datapath, modality, number_site, size_crop, nested)

    for i, conf in enumerate(networks_config):
        test_dicemetric = []
        valid_dicemetric = []
        for rep in range(num_repetitions):
            print(f"{networks_name[i]} iteration {rep+1}")
            print(conf)
            
            conf["transfo"] = transfo
            conf["partitions_paths"]=partitions_paths
                
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

def benchmark_models(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  number_site=3, size_crop=100, nested=True):

    #fetch the files paths, create the data loading/augmentation routines
    partitions_paths, transfo = dataPreprocessing(datapath, modality, number_site, size_crop, nested)

    for i, conf in enumerate(networks_config):
        test_dicemetric = []
        valid_dicemetric = []
        print(f"{networks_name[i]}")
        rep=0    
        conf["transfo"] = transfo
        conf["partitions_paths"]=partitions_paths
            
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

        print("Score for model minimizing dice loss during training over validation set")
        # compute validation and test dice score using the model with the smallest dice loss
        valid_dicemetric_benchLOSS = network.full_volume_metric(dataset="valid", network="best", benchmark_metric="diceloss", save_pred=True, verbose=False)[0]
        test_dicemetric_benchLOSS = network.full_volume_metric(dataset="test", network="best",  benchmark_metric="diceloss", save_pred=True, verbose=False)[0]

        print(f"{networks_name[i]} valid avg dice: {valid_dicemetric_benchLOSS}")
        print(f"{networks_name[i]} test avg dice: {test_dicemetric_benchLOSS}")

        print("Score for model maximizing dice score during training over validation set")
        # compute validation and test dice score using the model with the largest dice score
        valid_dicemetric_benchSCORE = network.full_volume_metric(dataset="valid", network="best", benchmark_metric="dicescore", save_pred=True, verbose=False)[0]
        test_dicemetric_benchSCORE = network.full_volume_metric(dataset="test", network="best", benchmark_metric="dicescore", save_pred=True, verbose=False)[0]

        print(f"{networks_name[i]} valid avg dice: {valid_dicemetric_benchSCORE}")
        print(f"{networks_name[i]} test avg dice: {test_dicemetric_benchSCORE}")


if __name__ == '__main__':
    #path = 'astral_fedem_dti_purged/'
    path = 'astral_fedem_dti/'
    #path = 'astral_fedem_dti_noempty/'
    #path = 'astral_fedem_v3/'

    #experience_name = "astral_no_empty_mask"
    experience_name = "full_dataset"
    modality="ADC"

    clients=["center1", "center2", "center3"]
    number_site=len(clients)

    default = {"g_epoch":50,
               "l_epoch":20,
               "g_lr":0.01,
               "l_lr":0.0001,
               "K":len(clients),
               "clients":clients,
               "suffix":"exp5",
               "val_interval":2,
               "modality":modality.lower(),
               "batch_size":8,
               'early_stop_limit':20,
               #all the parameters required to use the "new" torchio dataloader, a lot more of data augmentation
               "use_torchio":False,
               "clamp_min":0,
               "clamp_max":4000,
               "patch_size":(128,128,1),
               "max_queue_length":16,
               "patches_per_volume":4,
               }

    #thres_lesion_vol indicate the minimum number of 1 label in the mask required to avoid elimination from the dataset
    check_dataset(path, number_site, dim=(144,144,42), delete=True, thres_neg_val=-1e-6, thres_lesion_vol=0)

    networks_config = []
    networks_name = []
    #for lr in np.linspace(1e-5, 1e-2, 5):
    for lr in [0.005985, 0.001694, 0.00994, 0.01164]:
        tmp = default.copy()
        tmp.update({"centralized":True, "l_lr":lr})
        networks_config.append(tmp)
        networks_name.append(f"{experience_name}_CENTRALIZED_lr{lr}_batch{tmp['batch_size']}_epoch{tmp['g_epoch']*tmp['l_epoch']}")

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
                                                number_site=number_site,
                                                size_crop=144,
                                                nested=False)
     
    benchmark_models(datapath=path,
                     num_repetitions=1,
                     networks_config=networks_config,
                     networks_name=networks_name,
                     exp_name=experience_name,
                     modality=modality,
                     number_site=number_site,
                     size_crop=144,
                     nested=False)
