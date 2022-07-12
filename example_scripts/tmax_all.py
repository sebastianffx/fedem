
from framework import Scaffold, FedAvg, FedRod, Fedem
from preprocessing import dataPreprocessing
from numpy import std, mean

path = '/str/data/ASAP/miccai22_data/isles/federated/'
networks_name = ["SCAFFOLD", "FEDAVG", "FEDBETA", "FEDROD"]
modality = 'Tmax'
clients=["center1", "center2", "center4"]
datapath = '/str/data/ASAP/miccai22_data/isles/federated/'
exp_name=None
batch_size = 2
K = len(clients)
local_epochs, global_epochs = 1, 3
local_lr, global_lr = 0.00932, 1.7 #np.sqrt(K)
num_repetitions = 5
number_site=4

partitions_paths, centers_data_loaders, all_test_loader, all_valid_loader, all_train_loader = dataPreprocessing(datapath, modality, number_site=4, batch_size =batch_size)        

default = {"g_epoch":2,
           "l_epoch":1,
           "g_lr":1.7,
           "l_lr":0.00932,
           "K":len(clients),
           "clients":clients,
           "modality": modality,
           "val_interval": 2,
           "valid_loader": all_valid_loader,
           "partitions_paths": partitions_paths
          }

scaff = default.copy()
scaff.update({'partitions_paths': partitions_paths, 'suffix': 'scaffold'})

fedavg = default.copy()
fedavg.update({"weighting_scheme":"FEDAVG",'suffix': 'fedavg'})

fedbeta = default.copy()
fedbeta.update({"weighting_scheme":"BETA",
                "beta_val":0.9, 'suffix': 'beta'})

fedrod = default.copy()
fedrod.update({'K': K, 'l_epoch': local_epochs, 'B': batch_size, 'g_epoch': global_epochs, 'clients': clients,
           'l_lr':local_lr, 'g_lr':global_lr, 'dataloader':centers_data_loaders, 'suffix': 'fedrod', 
           'scaffold_controls': False, 'fed_rod':True, 'valid_loader': all_valid_loader, 'partitions_paths': partitions_paths})

network = FedRod(fedrod)

networks_config = [scaff, fedavg, fedbeta, fedrod]

tmp_test = []
tmp_valid = []
for i, conf in enumerate(networks_config):
    test_dicemetric = []
    valid_dicemetric = []
    for rep in range(num_repetitions):
        print(f"{networks_name[i]} iteration {rep+1}")
        #create new loaders for each repetition
        _, centers_data_loaders, all_test_loader, all_valid_loader, all_train_loader = dataPreprocessing(datapath, modality, number_site, batch_size)
        conf["dataloader"]=centers_data_loaders
        #add number to differentiate replicates
        if exp_name!=None:
            conf["suffix"]="_"+exp_name+"_"+str(rep)
        else:
            conf["suffix"]+="_"+str(rep)
        
        print(conf['suffix'])
        if "scaff" in conf["suffix"]:
            print("Training with SCAFFOLD")
            network = Scaffold(conf)
        if "fedrod" in conf["suffix"]:
            print("Training with FedRod")
            network = FedRod(conf)
        elif 'weighting_scheme' in conf.keys():
            print("Training FedAvg with " + conf['weighting_scheme'])
            network = FedAvg(conf)

        network.train_server(conf['g_epoch'], conf['l_epoch'], conf['g_lr'], conf['l_lr'])

        valid_dicemetric.append(network.test(all_valid_loader, test=False))
        test_dicemetric.append(network.global_test_cycle())

    tmp_valid.append(valid_dicemetric)
    tmp_test.append(test_dicemetric)

for k, (valid_metrics, test_metrics) in enumerate(zip(tmp_valid, tmp_test)):
    print(f"{networks_name[k]} valid avg dice: {mean(valid_metrics)} std: {std(valid_metrics)}")
    print(f"{networks_name[k]} test avg dice: {mean(test_metrics)} std: {std(test_metrics)}")
print("Finished training!")
test_metrics