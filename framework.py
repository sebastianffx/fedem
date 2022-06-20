import os
import copy
import itertools
from pandas import options
import torch
import random
import monai
import numpy as np
import nibabel as nib

from network import UNet_custom
from monai.metrics import DiceMetric
from torch.optim import Optimizer, Adam
from preprocessing import generate_loaders, torchio_generate_loaders, torchio_create_test_transfo
from torch.utils.tensorboard import SummaryWriter
from weighting_schemes import average_weights, average_weights_beta, average_weights_softmax

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    EnsureType,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

print(f"Using {device} as backend")

class Fedem:
    def __init__(self, options):
        self.options = options

        #routine to convert U-Net output to segmentation mask
        self.post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        if self.options["use_torchio"]:
            self.dataloaders, self.all_test_loader, self.all_valid_loader, self.all_train_loader = torchio_generate_loaders(partitions_paths=options["partitions_paths"],
                                                                                                                            batch_size=self.options["batch_size"],
                                                                                                                            clamp_min=self.options["clamp_min"],
                                                                                                                            clamp_max=self.options["clamp_max"],
                                                                                                                            padding=self.options["padding"],
                                                                                                                            patch_size=self.options["patch_size"],
                                                                                                                            max_queue_length=self.options["max_queue_length"],
                                                                                                                            patches_per_volume=self.options["patches_per_volume"],
                                                                                                                            no_deformation=self.options["no_deformation"],
                                                                                                                            partitions_paths_add_mod=self.options["partitions_paths_add_mod"]
                                                                                                                            )

        if self.options["use_test_augm"]:
            self.options["test_time_augm"] = torchio_create_test_transfo()

    def train_server(self, global_epoch, local_epoch, global_lr, local_lr, early_stop_limit=-1):
        metric_values = list()
        best_metric = -1
        best_metric_epoch = -1
        index = [0,1,2]
        for cur_epoch in range(global_epoch):
            print("*** global_epoch:", cur_epoch+1, "***")

            if not self.options["use_torchio"]:
                #recreate the dataloader at each epoch to resample the slices and apply the data augmentation
                self.dataloaders, _, _, _ = generate_loaders(self.partitions_paths, self.options["transfo"], self.options["batch_size"])

            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index, local_epoch, local_lr, cur_epoch)
            # aggregation
            self.aggregation(index, global_lr)

            #Evaluation on validation (full volume) and saving model if needed
            if (cur_epoch + 1) % self.options['val_interval'] == 0:
                epoch_valid_dice_score, epoch_valid_dice_loss = self.full_volume_metric(dataset="valid", network="self", save_pred=False)
                if epoch_valid_dice_score > best_metric:
                    best_metric = epoch_valid_dice_score
                    best_metric_epoch = cur_epoch+1

                    torch.save(self.nn.state_dict(), os.path.join(".", "models", self.options["network_name"]+"_"+self.options['modality']+'_'+self.options['suffix']+'_best_metric_model_segmentation2d_array.pth'))
                    print("saved new best metric model (according to DICE SCORE)")

                #early stopping implementation; if the validation dice score don't change after X consecutive round, stop the training
                if early_stop_limit > 0:
                    if np.abs(early_stop_val - epoch_valid_dice_loss) < 1e-5:
                        early_stop_count += 1
                        if early_stop_count >= early_stop_limit:
                            print(f"Early stopping, the model has converged/diverged and the loss is constant for the last {early_stop_limit} epochs")
                            return self.nn
                    else:
                        early_stop_val = epoch_valid_dice_loss
                        early_stop_count = 0
        return self.nn

    def validation(self,index):        
        return NotImplementedError
    
    def client_update(self, index, local_epoch, local_lr, cur_epoch):
        tmp=0
        #round loss is assumed to be the generic model loss 
        for i in index:
            if "fedrod" in self.options.keys():
                self.nns[i], round_loss, round_loss_personalized_m = self.train(self.nns[i], self.dataloaders[i][0], local_epoch, local_lr)            
            else:
                self.nns[i], round_loss = self.train(self.nns[i], self.dataloaders[i][0], local_epoch, local_lr)
            print(self.nns[i].name+" loss:", round_loss)
            self.writer.add_scalar('training loss '+self.nns[i].name, round_loss, cur_epoch)
            tmp+=round_loss
            
        tmp/=len(index)
        self.writer.add_scalar('avg training loss', tmp, cur_epoch)

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()


    def test(self, dataloader_test, test=True, model_path=None):
        model=self.nn

        if model_path != None:
            print("Loading model weights: ")
            print(model_path)
            checkpoint = torch.load(model_path)
            #carefull, this overwrite the current model
            model.load_state_dict(checkpoint)

        model.eval()
        pred = []
        y = []
        dice_metric.reset()
        for test_data in dataloader_test:
            with torch.no_grad():                
                #print(test_data[0].shape)
                test_img, test_label = test_data[0][:,:,:,:,0].to(device), test_data[1][:,:,:,:,0].to(device)
                test_pred = model(test_img)
                #apply sigmoid and thresholding to convert U-Net output to segmentation mask
                test_pred = self.post_pred(test_pred) #This assumes one slice in the last dim
                dice_metric(y_pred=test_pred, y=test_label)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        print('validation set dice score:', metric)
        if test:
            self.writer.add_scalar('test dice metric', metric)
        else:
            self.writer.add_scalar('validation dice metric', metric)
        return metric

    def aggregation():
        raise NotImplementedError

    def train():
        raise NotImplementedError
    
    def full_volume_metric(self, dataset, network="best", benchmark_metric="dicescore", save_pred=False, verbose=True):
        """ Compute test metric for full volume of the test set

            network : if "best", the best model (dice loss on validation set) will be loaded and overwrite the current model
        """
        if save_pred:
            os.makedirs(os.path.join(".", "output_viz", self.options["network_name"]), exist_ok=True)

        if dataset=="test":
            dataset_loader = self.all_test_loader
        elif dataset=="valid":
            dataset_loader = self.all_valid_loader
        else:
            print("invalid dataset type, possible value are train, valid and test")
            return -1

        model = self.nn
        if network=="best":
            if benchmark_metric == "diceloss":
                model_path = os.path.join(".", "models", self.options["network_name"]+"_"+self.options['modality']+'_'+self.options['suffix']+'_best_metric_model_segmentation2d_array.pth')
            elif benchmark_metric == "dicescore":
                model_path = os.path.join(".", "models", self.options["network_name"]+"_"+self.options['modality']+'_'+self.options['suffix']+'_best_DICE_model_segmentation2d_array.pth')
            else:
                model_path = ""
                print("option for benchmarking metric is not valid")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint)

            if verbose:
                print("Loading best validation model weights: ")
                print(model_path)

        elif network=="self":
            print("Using current network weights")
        else:
            print("Network weights to load are unclear")
            return None

        model.eval()

        loss_function = monai.losses.DiceLoss(sigmoid=True)

        if self.options['modality'] =='cbf':
            max_intensity = 1200
        if self.options['modality'] =='cbv':
            max_intensity = 200
        if self.options['modality'] =='tmax' or self.options['modality'] =='mtt':
            max_intensity = 30
        if self.options['modality'] =='adc':
            max_intensity = 4000

        holder_dicemetric = []
        holder_diceloss = []
        dice_metric.reset()

        #### TMP : used to test the best approach to average the output
        dice_metric_augm = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        holder_dicemetric_augm = []
        ### END TMP

        #during validation and testing, the batch_data size should be 1, last dimension is number of slice in original volume
        for batch_data in dataset_loader: 
            inputs, labels = batch_data[self.options['modality']]['data'][:,:,:,:].to(device),batch_data['label']['data'][:,:,:,:].to(device)

            if self.options["use_test_augm"] and dataset=="test":
                #apply the transformation on the entire 3D volume, to avoid transfer between cpu and gpu for each slice
                #computationnaly faster, might just require more memory but since bact size is equal to 1, should be ok
                test_time_images = [augm(inputs.clone().cpu()[0,:,:,:,:]).to(device) for augm in self.options["test_time_augm"]]
                inverse_test_augm = [augm.inverse() for augm in self.options["test_time_augm"]]
                augm_pred_holder = []

            #raw_pred_holder = []
            post_pred_holder = []
            loss_volume= []
            #iterate over all the slices present in the volume
            for slice_selected in range(inputs.shape[-1]):
                out = model(inputs[:,:,:,:,slice_selected])
                
                #compute loss between output and label (loss function applies the sigmoid function itself)
                loss_volume.append(loss_function(input=out,
                                                 target=labels[:,:,:,:,slice_selected]
                                                ).item()
                                  )

                #saving the raw output of the network
                #raw_pred_holder.append(out[0,0,:,:].detach().cpu().numpy())
                #apply sigmoid then activation threshold to obtain a discrete segmentation mask
                pred = self.post_pred(out)
                #compute dice score between the processed prediction and the labels (single slice)
                dice_metric(pred,labels[:,:,:,:,slice_selected])
                #save the prediction slice to rebuild a 3D prediction volume
                post_pred_holder.append(pred[0,0,:,:].cpu().numpy())

                #perform test-time augmentation slice-wise
                if self.options["use_test_augm"] and dataset=="test":
                    #initialized with the original image output (before/after post_pred routine)
                    augm_preds = [pred] #pred is already on the device
                    for augmented_test_img, inverse_augm in zip(test_time_images, inverse_test_augm):
                        augm_out = model(augmented_test_img[None,:,:,:,slice_selected])
                        augm_out_inv = inverse_augm(augm_out.detach().cpu()) #happens on cpu

                        #would probably be good to manually re-check that the inverse augmentation are well applied

                        #apply sigmoid and threshold BEFORE averaging
                        augm_preds.append(self.post_pred(augm_out_inv).to(device))
                        #tried applying sigmoid and threshold AFTER averaging UNet output but the negative values overcome the positive prior to the sigmoid

                    #average must discretized, using a simple threshold at 0.5
                    avg_augm_pred = torch.mean(torch.stack(augm_preds, dim=0), dim=0).to(device) # stack into X, 1, 1, 144, 144, mean into 1, 1, 144, 144
                    avg_augm_pred = avg_augm_pred >= self.options["test_augm_threshold"] #threshold for positive labeling after augmentation prediction avg
                    avg_augm_pred = avg_augm_pred.int() #convert bool to int

                    dice_metric_augm(avg_augm_pred, labels[:,:,:,:,slice_selected])

                    augm_pred_holder.append(avg_augm_pred[0,0,:,:].cpu().numpy())

            if save_pred:
                affine = batch_data['label']['affine'][0,:,:].detach().cpu().numpy()
                filestem = batch_data['label']['stem'][0]
                #nib.save(nib.Nifti1Image(np.stack(raw_pred_holder, axis=-1), affine), os.path.join(".", "output_viz", self.options["network_name"], filestem.replace("_msk", "_raw_segpred_"+benchmark_metric+".nii.gz")))
                nib.save(nib.Nifti1Image(np.stack(post_pred_holder, axis=-1), affine), os.path.join(".", "output_viz", self.options["network_name"], filestem.replace("_msk", "_post_segpred_"+benchmark_metric+".nii.gz")))
                if self.options["use_test_augm"] and dataset=="test":
                    nib.save(nib.Nifti1Image(np.stack(augm_pred_holder, axis=-1), affine), os.path.join(".", "output_viz", self.options["network_name"], filestem.replace("_msk", "_augm_segpred_"+benchmark_metric+".nii.gz")))

            #retain each volume scores (dice loss and dice score)
            holder_dicemetric.append(dice_metric.aggregate().item()) #average per volume
            holder_diceloss.append(np.mean(loss_volume)) #average per volume
            # reset the status for next computation round
            dice_metric.reset()

            if self.options["use_test_augm"] and dataset=="test":
                holder_dicemetric_augm.append(dice_metric_augm.aggregate().item())
                dice_metric_augm.reset()

        #print average over all the volumes
        if verbose:
            print(f"Global (all sites, all slices) {dataset} DICE LOSS :", np.round(np.mean(holder_diceloss),4))
            print(f"Global (all sites, all slices) {dataset} DICE SCORE :", np.round(np.mean(holder_dicemetric),4))
            if self.options["use_test_augm"] and dataset=="test":
                print(f"Global (all sites, all slices) {dataset} DICE SCORE (test-augm):", np.round(np.mean(holder_dicemetric_augm),4))
        return np.mean(holder_dicemetric), np.mean(holder_diceloss)

class FedAvg(Fedem):
    def __init__(self, options):
        super(FedAvg, self).__init__(options)

        self.weighting_scheme = options['weighting_scheme']
        self.writer = SummaryWriter(f"runs/llr{options['l_lr']}_glr{options['g_lr']}_le{options['l_epoch']}_ge{options['g_epoch']}_{options['K']}sites_"+options['weighting_scheme']+options['suffix'])


        if options['weighting_scheme'] == 'BETA':
            self.beta_val = options['beta_val']
            #extract length of trianing loader for each site
            self.trainloaders_lengths = [len(ldtr[0]) for ldtr in self.dataloaders]
        elif options['weighting_scheme'] == 'SOFTMAX':
            #extract length of trianing loader for each site
            self.trainloaders_lengths = [len(ldtr[0]) for ldtr in self.dataloaders]
        
        #server model
        self.nn = UNet_custom(spatial_dims=2,
                              in_channels=1,
                              out_channels=1,
                              channels=(16, 32, 64, 128),
                              strides=(2, 2, 2),
                              kernel_size = (3,3),
                              num_res_units=2,
                              name='server',
                              scaff=False).to(device)
        
        #create clients
        self.nns = []
        for i in range(len(options['clients'])):
            temp = copy.deepcopy(self.nn)
            temp.name = options['clients'][i]
            self.nns.append(temp)

    def aggregation(self, index, global_lr):
        client_weights=[]

        #would be possible to use sampling here
        for client_nn in self.nns:
            client_weights.append(copy.deepcopy(client_nn.state_dict()))

        #Agregating the weights with the selected weighting scheme
        if self.weighting_scheme =='FEDAVG':
            global_weights = average_weights(client_weights)
        if self.weighting_scheme =='BETA':
            sub_trainloaders_lengths = [self.trainloaders_lengths[idx] for idx in index]
            global_weights = average_weights_beta(client_weights,sub_trainloaders_lengths,self.beta_val)
        if self.weighting_scheme =='SOFTMAX':
            sub_trainloaders_lengths = [self.trainloaders_lengths[idx] for idx in index]
            global_weights = average_weights_softmax(client_weights,sub_trainloaders_lengths)

        # Update global weights with the averaged model weights.
        self.nn.load_state_dict(global_weights)

    def train(self, ann, dataloader_train, local_epoch, local_lr):
        #train client to train mode
        ann.train()
        ann.len = len(dataloader_train)
                
        loss_function = monai.losses.DiceLoss(sigmoid=True)
        optimizer = Adam(ann.parameters(), local_lr)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
                if self.options["use_torchio"]:
                    inputs, labels = batch_data[self.options['modality']]['data'][:,:,:,:,0].to(device),batch_data['label']['data'][:,:,:,:,0].to(device)
                else:
                    inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
                y_pred = ann(inputs)
                loss = loss_function(y_pred, labels)
                optimizer.zero_grad()        
                loss.backward()
                optimizer.step()
                            
        return ann, loss.item()

class Scaffold(Fedem):
    def __init__(self, options):
        super(Scaffold, self).__init__(options)

        self.writer = SummaryWriter(f"runs/llr{options['l_lr']}_glr{options['g_lr']}_le{options['l_epoch']}_ge{options['g_epoch']}_{options['K']}sites_"+"SCAFFOLD"+options['suffix'])

        self.K = options['K']
        
        #server model
        self.nn = UNet_custom(spatial_dims=2,
                             in_channels=1,
                             out_channels=1,
                             channels=(16, 32, 64, 128),
                             strides=(2, 2, 2),
                             kernel_size = (3,3),
                             num_res_units=2,
                             name='server',
                             scaff=True).to(device)
        
        #control variables
        for k, v in self.nn.named_parameters():
            self.nn.control[k] = torch.zeros_like(v.data)
            self.nn.delta_control[k] = torch.zeros_like(v.data)
            self.nn.delta_y[k] = torch.zeros_like(v.data)
        
        #clients
        self.nns = []
        for i in range(len(options['clients'])):
            temp = copy.deepcopy(self.nn)
            temp.name = options['clients'][i]
            temp.control       = copy.deepcopy(self.nn.control)  # ci
            temp.delta_control = copy.deepcopy(self.nn.delta_control)  # ci
            temp.delta_y       = copy.deepcopy(self.nn.delta_y)
            self.nns.append(temp)

    def aggregation(self, index, global_lr, **kwargs):
        s = 0.0
        for j in index:
            # normal
            s += self.nns[j].len
        # compute
        x = {}
        c = {}
        # init
        for k, v in self.nns[0].named_parameters():
            x[k] = torch.zeros_like(v.data)
            c[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                x[k] += self.nns[j].delta_y[k] / len(index)  # averaging
                c[k] += self.nns[j].delta_control[k] / len(index)  # averaging

        # update x and c
        for k, v in self.nn.named_parameters():
            v.data += x[k].data*global_lr
            self.nn.control[k].data += c[k].data * (len(index) / self.K)

    def train(self, ann, dataloader_train, local_epoch, local_lr):
        #train client to train mode
        ann.train()
        ann.len = len(dataloader_train)
                
        x = copy.deepcopy(ann)
        loss_function = monai.losses.DiceLoss(sigmoid=True)
        optimizer = ScaffoldOptimizer(ann.parameters(), lr=local_lr, weight_decay=1e-4)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
                if self.options["use_torchio"]:
                    inputs, labels = batch_data[self.options['modality']]['data'][:,:,:,:,0].to(device),batch_data['label']['data'][:,:,:,:,0].to(device)
                else:
                    inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
                y_pred = ann(inputs)
                loss = loss_function(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()          
                optimizer.step(self.nn.control, ann.control) #performing SGD on the control variables
                            
        # update c
        # c+ <- ci - c + 1/(E * lr) * (x-yi)
        temp = {}
        for k, v in ann.named_parameters():
            temp[k] = v.data.clone()
        for k, v in x.named_parameters():
            ann.control[k] = ann.control[k] - self.nn.control[k] + (v.data - temp[k]) / (local_epoch * local_lr)
            ann.delta_y[k] = temp[k] - v.data
            ann.delta_control[k] = ann.control[k] - x.control[k]
        return ann, loss.item()

    def global_test(self, aggreg_dataloader_test):
        model = self.nn
        model.eval()
        
        #test the global model on each individual dataloader
        for k, client in enumerate(self.nns):
            print("testing on", client.name, "dataloader")
            test(model, self.dataloaders[k][2])
        
        #test the global model on aggregated dataloaders
        print("testing on all the data")
        test(model, aggreg_dataloader_test)

class FedRod(Fedem):
    def __init__(self, options):
        super(FedRod, self).__init__(options)
        self.writer = SummaryWriter(f"runs/llr{options['l_lr']}_glr{options['g_lr']}_le{options['l_epoch']}_ge{options['g_epoch']}_{options['K']}sites_"+"FEDROD"+options['suffix'])
        self.K = options['K']
        #self.name_encoder_layers = ["model.0", "model.1.submodule.0", "model.1.submodule.1.submodule.2.0",
        #                            "model.1.submodule.1.submodule.0", "model.1.submodule.1.submodule.1"]
        self.name_encoder_layers = ["model.0.conv.unit0.conv.weight",
                                    "model.0.conv.unit0.conv.bias",
                                    "model.0.conv.unit0.adn.A.weight",
                                    "model.0.conv.unit1.conv.weight",
                                    "model.0.conv.unit1.conv.bias",
                                    "model.0.conv.unit1.adn.A.weight",
                                    "model.0.residual.weight",
                                    "model.0.residual.bias",
                                    "model.1.submodule.0.conv.unit0.conv.weight",
                                    "model.1.submodule.0.conv.unit0.conv.bias",
                                    "model.1.submodule.0.conv.unit0.adn.A.weight",
                                    "model.1.submodule.0.conv.unit1.conv.weight",
                                    "model.1.submodule.0.conv.unit1.conv.bias",
                                    "model.1.submodule.0.conv.unit1.adn.A.weight",
                                    "model.1.submodule.0.residual.weight",
                                    "model.1.submodule.0.residual.bias",
                                    "model.1.submodule.1.submodule.0.conv.unit0.conv.weight",
                                    "model.1.submodule.1.submodule.0.conv.unit0.conv.bias",
                                    "model.1.submodule.1.submodule.0.conv.unit0.adn.A.weight",
                                    "model.1.submodule.1.submodule.0.conv.unit1.conv.weight",
                                    "model.1.submodule.1.submodule.0.conv.unit1.conv.bias",
                                    "model.1.submodule.1.submodule.0.conv.unit1.adn.A.weight",
                                    "model.1.submodule.1.submodule.0.residual.weight",
                                    "model.1.submodule.1.submodule.0.residual.bias",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.weight",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit0.conv.bias",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit0.adn.A.weight",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.weight",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit1.conv.bias",
                                    "model.1.submodule.1.submodule.1.submodule.conv.unit1.adn.A.weight",
                                    "model.1.submodule.1.submodule.1.submodule.residual.weight",
                                    "model.1.submodule.1.submodule.1.submodule.residual.bias",
                                    "model.1.submodule.1.submodule.2.0.conv.weight",
                                    "model.1.submodule.1.submodule.2.0.conv.bias",
                                    "model.1.submodule.1.submodule.2.0.adn.A.weight",
                                    "model.1.submodule.1.submodule.2.1.conv.unit0.conv.weight",
                                    "model.1.submodule.1.submodule.2.1.conv.unit0.conv.bias",
                                    "model.1.submodule.1.submodule.2.1.conv.unit0.adn.A.weight",
                                    "model.1.submodule.2.0.conv.weight",
                                    "model.1.submodule.2.0.conv.bias",
                                    "model.1.submodule.2.0.adn.A.weight",
                                    "model.1.submodule.2.1.conv.unit0.conv.weight",
                                    "model.1.submodule.2.1.conv.unit0.conv.bias",
                                    "model.1.submodule.2.1.conv.unit0.adn.A.weight",
                                    'model.1.submodule.1.submodule.2.1']

        
        self.name_decoder_layers  = ["model.2.0.conv.weight",
                                    "model.2.0.conv.bias",
                                    "model.2.0.adn.A.weight",
                                    "model.2.1.conv.unit0.conv.weight",
                                    "model.2.1.conv.unit0.conv.bias"]

        self.trainloaders_lengths = [len(ldtr[0]) for ldtr in self.dataloaders]
        print(self.trainloaders_lengths)
        #server model
        self.nn = UNet_custom(spatial_dims=2,
                             in_channels=1,
                             out_channels=1,
                             channels=(16, 32, 64, 128),
                             strides=(2, 2, 2),
                             kernel_size = (3,3),
                             num_res_units=2,
                             name='server',
                             scaff=False,
                             fed_rod=True).to(device)
        
        
        #Global encoder - decoder (inlcuding personalized) layers init
        for k, v in self.nn.named_parameters():

            for enc_layer in self.name_encoder_layers:
                if enc_layer == k:
                    self.nn.encoder_generic[k] = copy.deepcopy(v.data)
            for dec_layer in self.name_decoder_layers:
                if dec_layer == k:
                    self.nn.decoder_generic[k] = copy.deepcopy(v.data)
                    self.nn.decoder_personalized[k] = torch.zeros(v.data.shape)
                    #self.nn.decoder_personalized[k] = copy.deepcopy(v.data)
                    
        #print(self.nn.decoder_generic)
        #clients of the federation
        self.nns = []
        for i in range(len(options['clients'])):
            temp = copy.deepcopy(self.nn)
            temp.name = options['clients'][i]
            temp.encoder_generic = copy.deepcopy(self.nn.encoder_generic)
            temp.decoder_generic = copy.deepcopy(self.nn.decoder_generic)
            temp.decoder_personalized = copy.deepcopy(self.nn.decoder_personalized)            
            self.nns.append(temp)
            
    

    def aggregation(self, index, global_lr, **kwargs):
        s = 0.0
        for j in index:
            # normal
            s += self.nns[j].len
        #print(self.trainloaders_lengths)   
        #print(sum(self.trainloaders_lengths)) 
        num_training_data_samples = np.sum([self.trainloaders_lengths[z] for z in index])        
        #print(num_training_data_samples)
        #print(index)
        # Agregating the generic encoder from clients encoders
        
        avg_weights_encoder = copy.deepcopy(self.nns[0].encoder_generic)    
        for k, v in self.nn.named_parameters():    
        #for key in avg_weights_encoder:
            if k in avg_weights_encoder.keys():
                for j in index[1:]:
                    for enc_layer in self.name_encoder_layers:
                        if enc_layer == k:
                            weight_contribution = self.trainloaders_lengths[j]/num_training_data_samples
                            avg_weights_encoder[k] += torch.mul(self.nns[j].encoder_generic[k], weight_contribution) #check other weightings here            
                v.data = torch.div(avg_weights_encoder[k], len(index))

        # Agregating the generic decoder from clients decoders
        avg_weights_decoder = copy.deepcopy(self.nns[0].decoder_generic)        
        for k, v in self.nn.named_parameters():    
            if k in avg_weights_decoder.keys():
                for j in index[1:]:
                    for dec_layer in self.name_decoder_layers:
                        if dec_layer == k:
                            weight_contribution = self.trainloaders_lengths[j]/num_training_data_samples
                            avg_weights_decoder[k] += torch.mul(self.nns[j].decoder_generic[k], weight_contribution) #check other weightings here            
                v.data = torch.div(avg_weights_decoder[k], len(index))


    def train(self, ann, dataloader_train, local_epoch, local_lr):
        #First the generic encoder-decoder are updated       
        ann.train()
        ann.len = len(dataloader_train)
                
        loss_function = monai.losses.DiceLoss(sigmoid=True)

        optimizer = torch.optim.Adam(ann.parameters(), lr=local_lr)
        loss_generic = loss_function(torch.tensor(np.zeros((1,10))), torch.tensor(np.zeros((1,10))))
        loss_personalized = loss_function(torch.tensor(np.zeros((1,10))), torch.tensor(np.zeros((1,10))))
        # for k, v in self.nn.named_parameters():
        # print(v.requires_grad, v.dtype, v.device, k)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
                #(1)Optimization of the Generic path here equation (8) of the paper
                for k, v in ann.named_parameters(): #Transfering data from the generic head is done in dispatch()
                    v.requires_grad = True #deriving gradients to all the generic layers
                
                if self.options["use_torchio"]:
                    inputs, labels = batch_data[self.options['modality']]['data'][:,:,:,:,0].to(device),batch_data['label']['data'][:,:,:,:,0].to(device)
                else:
                    inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
                y_pred_generic = ann(inputs)
                loss_generic   = loss_function(y_pred_generic, labels)
                optimizer.zero_grad()
                loss_generic.backward()
                optimizer.step()
                #Keeping the updated generic parameters
                for k, v in ann.named_parameters():
                    for enc_layer in self.name_encoder_layers:
                        if enc_layer == k:
                            ann.encoder_generic[k] = copy.deepcopy(v.data).to(device) #Keeping the generic encoder data

                for k, v in ann.named_parameters():
                    for dec_layer in self.name_decoder_layers:
                        if dec_layer == k:
                            ann.decoder_generic[k] = copy.deepcopy(v.data).to(device) #Keeping the generic decoder data
                
                #(2)Optimization of the Perzonalized path here equation (9) of the paper
                #(3) Keeping the generic output to add it later to the personalized
                output_generic = 0
                if device.type =='cuda':
                    output_generic = copy.deepcopy(y_pred_generic.detach())
                if device.type =='cpu':
                    output_generic = copy.deepcopy(y_pred_generic.detach().numpy())

                #print("OUTPUT SHAPE")
                #print(output_generic.shape)

                for k,v in ann.named_parameters():
                    for enc_layer_name in self.name_encoder_layers:
                        if enc_layer_name == k:
                            v.requires_grad = False

                for k, v in ann.named_parameters():
                    for dec_layer in self.name_decoder_layers:
                        if dec_layer == k:
                            v.data = ann.decoder_personalized[k].data.to(device) #"Swapping the heads"
                            v.requires_grad = True #Deriving gradients only wrt to the personalized head
               

                #output_personalized = ann(inputs) + torch.tensor(output_generic).to(device) #regularized personalized output
                output_personalized = ann(inputs) + output_generic.clone().detach().requires_grad_(requires_grad=True).to(device) #changed to stop a torch warning
                loss_personalized = loss_function(output_personalized, labels)
                optimizer.zero_grad()
                loss_personalized.backward()
                optimizer.step()

                for k, v in ann.named_parameters():
                    for dec_layer in self.name_decoder_layers:
                        if dec_layer == k:
                            ann.decoder_personalized[k].data = copy.deepcopy(v.data) #Saving the personalized head values                            
        return ann, loss_generic.item(), loss_personalized.item()

    def global_test(self, aggreg_dataloader_test):
        model = self.nn
        model.eval()
        
        #test the global model on each individual dataloader
        for k, client in enumerate(self.nns):
            print("testing on", client.name, "dataloader")
            test(model, self.dataloaders[k][2])
        
        #test the global model on aggregated dataloaders
        print("testing on all the data")
        test(model, aggreg_dataloader_test)
        
class Centralized(Fedem):
    def __init__(self, options):
        super(Centralized, self).__init__(options)
        self.nn = UNet_custom(spatial_dims=2,
                            in_channels=1,
                            out_channels=1,
                            channels=(16, 32, 64, 128),
                            strides=(2, 2, 2),
                            kernel_size = (3,3),
                            num_res_units=2,
                            name='centralized').to(device)

        self.writer = SummaryWriter(f"runs/llr{options['l_lr']}_glr{options['g_lr']}_le{options['l_epoch']}_ge{options['g_epoch']}_{options['K']}sites_"+options["network_name"]+options['suffix'])

    #overwrite the superclass method since there are no client models
    def train_server(self, global_epoch, local_epoch, global_lr, local_lr, save_train_pred=False, early_stop_limit=-1):
        metric_values = list()
        best_metric = -1
        best_metric_epoch = -1

        early_stop_val = 0
        early_stop_count = 0

        if self.options["loss_fun"] == "diceloss":
            print("Using DiceLoss as loss function")
            loss_function = monai.losses.DiceLoss(sigmoid=True) #should see the impact of batch=True
        else:
            print("Using DiceLoss + CE as loss function")
            loss_function = monai.losses.DiceCELoss(include_background=True, sigmoid=True, reduction='mean',
                                                    batch=True, ce_weight=None, 
                                                    lambda_dice=self.options["hybrid_loss_weights"][0], 
                                                    lambda_ce=self.options["hybrid_loss_weights"][1]
                                                    )

        optimizer = torch.optim.Adam(self.nn.parameters(), lr=local_lr)

        #multiply global and local epoch to have similar conditions
        for cur_epoch in range(global_epoch*local_epoch):
            print("*** epoch:", cur_epoch+1, "***")

            self.nn.train()
            
            if not self.options["use_torchio"]:
                #create new loaders for each repetition, to force the sampling of new slices and application of data augmentation
                _, _, _, self.all_train_loader = generate_loaders(self.partitions_paths, self.options["transfo"], self.options["batch_size"])

            epoch_loss = 0
            step = 0
            dice_metric.reset()

            for batch_data in self.all_train_loader:
                for k, v in self.nn.named_parameters():
                    v.requires_grad = True
                
                step += 1
                if self.options["use_torchio"]:
                    inputs, labels = batch_data[self.options['modality']]['data'][:,:,:,:,0].to(device),batch_data['label']['data'][:,:,:,:,0].to(device)
                else:
                    inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)

                y_pred_generic = self.nn(inputs)
                loss = loss_function(input=y_pred_generic, target=labels) #average over the batch after computing it for each slice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #apply sigmoid and threshold, since the loss function apply sigmoid to the output
                test_pred = self.post_pred(y_pred_generic)
                dice_metric(y_pred=test_pred, y=labels)

                if save_train_pred and (cur_epoch+1)%5==0 and labels[0,0,:,:].detach().cpu().numpy().sum() > 0:
                    #saving the slice of the first element of each batch during training, with and without prediction post-processing (sigmoid + threshold)
                    nib.save(nib.Nifti1Image(inputs[0,0,:,:].detach().cpu().numpy(), None), os.path.join(".", "output_viz", "viz_input_epoch"+str(cur_epoch+1)+"_adc.nii.gz"))
                    nib.save(nib.Nifti1Image(test_pred[0,0,:,:].detach().cpu().numpy(), None), os.path.join(".", "output_viz", "viz_input_epoch"+str(cur_epoch+1)+"_postpred.nii.gz"))
                    nib.save(nib.Nifti1Image(y_pred_generic[0,0,:,:].detach().cpu().numpy(), None), os.path.join(".", "output_viz", "viz_input_epoch"+str(cur_epoch+1)+"_rawpred.nii.gz"))
                    nib.save(nib.Nifti1Image(labels[0,0,:,:].detach().cpu().numpy(), None), os.path.join(".", "output_viz", "viz_input_epoch"+str(cur_epoch+1)+"_label.nii.gz"))
            
            #aggregate makes the average of the values, nan are ignored.
            print("training dice SCORE : {:.4f}".format(dice_metric.aggregate().item()))
            self.writer.add_scalar('avg training dice score', dice_metric.aggregate().item(), cur_epoch)
            print("training dice LOSS : {:.4f}".format(epoch_loss/step))
            self.writer.add_scalar('avg training dice loss', epoch_loss/step, cur_epoch)

            #Evaluation on validation and saving model if needed, on full volume
            if (cur_epoch + 1) % self.options['val_interval'] == 0:
                epoch_valid_dice_score, epoch_valid_dice_loss = self.full_volume_metric(dataset="valid", network="self", save_pred=False)

                #using dice score to save best model
                if epoch_valid_dice_score > best_metric:
                    best_metric = epoch_valid_dice_score
                    best_metric_epoch = cur_epoch+1

                    torch.save(self.nn.state_dict(), os.path.join(".", "models", self.options["network_name"]+"_"+self.options['modality']+'_'+self.options['suffix']+'_best_DICE_model_segmentation2d_array.pth'))
                    print("saved new best model (according to DICE SCORE)")

                print("validation dice SCORE : {:.4f}, best valid. dice SCORE: {:.4f} at epoch {}".format(
                    epoch_valid_dice_score, best_metric, best_metric_epoch)
                     )
                self.writer.add_scalar("avg validation dice score", epoch_valid_dice_score, cur_epoch)

                print("validation dice LOSS: {:.4f}".format(
                    epoch_valid_dice_loss)
                     )
                self.writer.add_scalar('avg validation loss', epoch_valid_dice_loss, cur_epoch)

                #early stopping implementation; if the validation dice score don't change after X consecutive round, stop the training
                if early_stop_limit > 0:
                    if np.abs(early_stop_val - epoch_valid_dice_loss) < 1e-5:
                        early_stop_count += 1
                        if early_stop_count >= early_stop_limit:
                            print(f"Early stopping, the model has converged/diverged and the loss is constant for the last {early_stop_limit} epochs")
                            return self.nn
                    else:
                        early_stop_val = epoch_valid_dice_loss
                        early_stop_count = 0

        ## DEBUG: save the prediction for the training set
        #self.full_volume_metric(dataset="training", network="self", save_pred=True)
        return self.nn

#optimizer
class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                #local learning rate
                p.data = p.data - dp.data * group['lr']
