from framework import Scaffold, FedAvg, FedRod, Fedem
from preprocessing import dataPreprocessing
from numpy import std, mean

import os
import nibabel as nb

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="ADC",
                  number_site=3, batch_size=2, size_crop=100, nested=True):
    tmp_test = []
    tmp_valid = []
    for i, conf in enumerate(networks_config):

        test_dicemetric = []
        valid_dicemetric = []
        for rep in range(num_repetitions):
            print(f"{networks_name[i]} iteration {rep+1}")

            #create new loaders for each repetition
            _, centers_data_loaders, all_test_loader, all_valid_loader = dataPreprocessing(datapath, modality, number_site, batch_size, size_crop, nested)
            conf["dataloader"]=centers_data_loaders
            conf["valid_loader"]=all_valid_loader
                
            #add number to differentiate replicates
            if exp_name!=None:
                conf["suffix"]="_"+exp_name+"_"+str(rep)
            else:
                conf["suffix"]="_"+str(rep)

            print("config for experiment", conf.keys())

            if "scaff" in conf.keys() and conf["scaff"]:
                network = Scaffold(conf)
            elif "FedRod" in conf.keys() and conf["fed_rod"]:
                network = FedRod(conf)
            elif 'weighting_scheme' in conf.keys():
                network = FedAvg(conf)
            else:
                print("missing argument for network type")

            network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'])

            valid_dicemetric.append(network.test(all_valid_loader, test=False))
            test_dicemetric.append(network.global_test_cycle())

        tmp_valid.append(valid_dicemetric)
        tmp_test.append(test_dicemetric)

    for k, (valid_metrics, test_metrics) in enumerate(zip(tmp_valid, tmp_test)):
        print(f"{networks_name[k]} valid avg dice: {mean(valid_metrics)} std: {std(valid_metrics)}")
        print(f"{networks_name[k]} test avg dice: {mean(test_metrics)} std: {std(test_metrics)}")

    return tmp_valid, tmp_test

def check_dataset(path, number_site, dim=(144,144,42)):
    for i in range(1,number_site+1):
        files_name=os.listdir(path+"train/")
        for f in files_name:
            tmp_shape = nb.load(path+"train/"+f).get_fdata().shape
            if tmp_shape != dim:
                print(path+"train/"+f, tmp_shape)

        files_name=os.listdir(path+"valid/")
        for f in files_name:
            tmp_shape = nb.load(path+"valid/"+f).get_fdata().shape
            if tmp_shape != dim:
                print(path+"valid/"+f, tmp_shape)

        files_name=os.listdir(path+"test/")
        for f in files_name:
            tmp_shape = nb.load(path+"train/"+f).get_fdata().shape
            if tmp_shape != dim:
                print(path+"test/"+f, tmp_shape)

if __name__ == '__main__':
    path = 'astral_fedem_v2/'
    modality="ADC"
    networks_name = ["SCAFFOLD", "FEDAVG", "FEDBETA"]

    clients=["center1", "center2", "center3"]
    number_site=len(clients)

    default = {"g_epoch":5,
               "l_epoch":5,
               "g_lr":1.7,
               "l_lr":0.00932,
               "K":len(clients),
               "clients":clients,
               "suffix":"exp1"
               }

    check_dataset(path, number_site, dim=(144,144,42))

    scaff = default.copy()
    scaff.update({"scaff":True})

    fedavg = default.copy()
    fedavg.update({"weighting_scheme":"FEDAVG"})

    fedbeta = default.copy()
    fedbeta.update({"weighting_scheme":"BETA",
                    "beta_val":0.9})

    networks_config = [scaff, fedavg, fedbeta]

    valid_metrics, test_metrics = runExperiment(datapath=path,
                                                num_repetitions=3,
                                                networks_config=networks_config,
                                                networks_name=networks_name,
                                                exp_name="test_astral",
                                                modality=modality,
                                                number_site=number_site,
                                                batch_size=2,
                                                size_crop=100,
                                                nested=False)