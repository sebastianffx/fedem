from framework import Scaffold, FedAvg
from preprocessing import dataPreprocessing
from numpy import std, mean

def runExperiment(datapath, num_repetitions, networks_config, networks_name, exp_name=None, modality="Tmax", number_site=4, batch_size=2):
    tmp = []
    for i, conf in enumerate(networks_config):

        test_dicemetric = []
        for rep in range(num_repetitions):
            print(f"{networks_name[i]} iteration {rep+1}")

            #create new loaders for each repetition
            _, centers_data_loaders, all_test_loader, _ = dataPreprocessing(datapath, modality, number_site, batch_size)
            conf["dataloader"]=centers_data_loaders
            
            #add number to differentiate replicates
            if exp_name!=None:
                conf["suffix"]="_"+exp_name+"_"+str(rep)
            else:
                conf["suffix"]="_"+str(rep)

            if "scaff" in conf.keys() and conf["scaff"]:
                network = Scaffold(conf)
            elif 'weighting_scheme' in conf.keys():
                network = FedAvg(conf)
            else:
                print("missing argument for network type")

            network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'])

            test_dicemetric.append(network.test(all_test_loader))

        tmp.append(test_dicemetric)

    for k, metrics in enumerate(tmp):
        print(f"{networks_name[k]} avg dice: {mean(metrics)} std: {std(metrics)}")

    return tmp