from framework import Scaffold, FedAvg, FedRod, Fedem, Centralized
from preprocessing import dataPreprocessing
from numpy import std, mean
import numpy as np

import os
import nibabel as nb

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  number_site=3, batch_size=2, size_crop=100, nested=True):

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
            network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'])

            #compute the validation dice metric of single slice
            valid_dicemetric.append(network.test(all_valid_loader, test=False))

            #compute test dice metric of full volume
            test_dicemetric.append(network.global_test_cycle())

        tmp_valid.append(valid_dicemetric)
        tmp_test.append(test_dicemetric)

    for k, (valid_metrics, test_metrics) in enumerate(zip(tmp_valid, tmp_test)):
        print(f"{networks_name[k]} valid avg dice: {mean(valid_metrics)} std: {std(valid_metrics)}")
        print(f"{networks_name[k]} test avg dice: {mean(test_metrics)} std: {std(test_metrics)}")

    return tmp_valid, tmp_test

def check_dataset(path, number_site, dim=(144,144,42), delete=True):
    bad_dim_files = []
    for i in range(1,number_site+1):
        files_name=os.listdir("./"+path+"center"+str(i)+"/train/")
        for f in files_name:
            tmp= nb.load("./"+path+"center"+str(i)+"/train/"+f).get_fdata()
            tmp_shape = tmp.shape
            if tmp_shape != dim:
                bad_dim_files.append("./"+path+"center"+str(i)+"/train/"+f)
                print("./"+path+"center"+str(i)+"/train/"+f, tmp_shape)
            if tmp.min()<-1e-6:
                print("./"+path+"center"+str(i)+"/train/"+f, "contains negative value")
            if np.isnan(tmp).sum() > 0:
                print("./"+path+"center"+str(i)+"/train/"+f, "contains NaN")

        files_name=os.listdir("./"+path+"center"+str(i)+"/valid/")
        for f in files_name:
            tmp= nb.load("./"+path+"center"+str(i)+"/valid/"+f).get_fdata()
            tmp_shape = tmp.shape
            if tmp_shape != dim:
                bad_dim_files.append("./"+path+"center"+str(i)+"/valid/"+f)
                print("./"+path+"center"+str(i)+"/valid/"+f, tmp_shape)
            if tmp.min()<-1e-6:
                print("./"+path+"center"+str(i)+"/valid/"+f, "contains negative value")
            if np.isnan(tmp).sum() > 0:
                print("./"+path+"center"+str(i)+"/valid/"+f, "contains NaN")

        files_name=os.listdir("./"+path+"center"+str(i)+"/test/")
        for f in files_name:
            tmp= nb.load("./"+path+"center"+str(i)+"/test/"+f).get_fdata()
            tmp_shape = tmp.shape
            if tmp_shape != dim:
                bad_dim_files.append("./"+path+"center"+str(i)+"/test/"+f)
                print("./"+path+"center"+str(i)+"/test/"+f, tmp_shape)
            if tmp.min()<-1e-6:
                print("./"+path+"center"+str(i)+"/test/"+f, "contains negative value")
            if np.isnan(tmp).sum() > 0:
                print("./"+path+"center"+str(i)+"/test/"+f, "contains NaN")

    if delete:
        for f in bad_dim_files:
            os.remove(f)
    else:
        print("TBD")
        # pad them with zeros? instead of deleting them?

if __name__ == '__main__':
    path = 'astral_fedem_dti_purged/'
    #path = 'astral_fedem_dti/'
    #path = 'astral_fedem_v3/'
    modality="ADC"

    clients=["center1", "center2", "center3"]
    number_site=len(clients)

    default = {"g_epoch":20,
               "l_epoch":10,
               "g_lr":0.01,
               "l_lr":0.0001,
               "K":len(clients),
               "clients":clients,
               "suffix":"exp5",
               "val_interval":2,
               "modality":modality,
               "batch_size":8
               }

    #check_dataset(path, number_site, dim=(144,144,42))

    networks_config = []
    networks_name = []
    #for lr in np.linspace(1e-5, 1e-2, 5):
    for lr in [0.005, 0.009, 0.01]:
        tmp = default.copy()
        tmp.update({"centralized":True, "l_lr":lr})
        networks_config.append(tmp)
        networks_name.append(f"CENTRALIZED_lr{lr}_batch{tmp['batch_size']}")

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
                                                exp_name="test_astral_2",
                                                modality=modality,
                                                number_site=number_site,
                                                batch_size=default["batch_size"],
                                                size_crop=144,
                                                nested=False)
