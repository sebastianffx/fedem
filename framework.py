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
from torch.utils.tensorboard import SummaryWriter
from weighting_schemes import average_weights, average_weights_beta, average_weights_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
print(device)

class Fedem:
    def __init__(self, options):
        #save all clients dataloader
        self.dataloaders = options['dataloader']
        self.valid_loader = options['valid_loader']
        self.options = options
        print(self.options['suffix'])

    def train_server(self, global_epoch, local_epoch, global_lr, local_lr):
        metric_values = list()
        best_metric = -1
        best_metric_epoch = -1
        index = [0,1,2]
        for cur_epoch in range(global_epoch):
            print("*** global_epoch:", cur_epoch+1, "***")

            #skiping center 2 as only 1 scan is available (train_loader sorted)

            #perform sampling here if desired

            # dispatch
            
            self.dispatch(index)
            # local updating
            self.client_update(index, local_epoch, local_lr, cur_epoch)
            # aggregation
            self.aggregation(index, global_lr)

            #Evaluation on validation and saving model if needed
            if (cur_epoch + 1) % self.options['val_interval'] == 0:
                best_metric,best_metric_epoch = self.global_validation_cycle(index,metric_values,cur_epoch,best_metric,best_metric_epoch)
        return self.nn

    def validation(self,index):        
        return NotImplementedError
    
    def client_update(self, index, local_epoch, local_lr, cur_epoch):
        tmp=0
        #round loss is assumed to be the generic model loss 
        for i in index:
            if "fedrod" in self.options['suffix']:
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
                #what is the purpose of the line below?
                test_pred =  test_pred>0.5 #This assumes one slice in the last dim
                dice_metric(y_pred=test_pred, y=test_label)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        print('dice:', metric)
        if test:
            self.writer.add_scalar('test dice metric', metric)
        else:
            self.writer.add_scalar('validation dice metric', metric)
        return metric

    def aggregation():
        raise NotImplementedError

    def train():
        raise NotImplementedError
    
    def global_validation_cycle(self,index,metric_values,cur_epoch, best_metric,best_metric_epoch):
        partitions_valid_imgs = [self.options['partitions_paths'][i][0][1] for i in range(len(self.options['partitions_paths']))]
        partitions_valid_lbls = [self.options['partitions_paths'][i][1][1] for i in range(len(self.options['partitions_paths']))]
        all_valid_paths  = list(itertools.chain.from_iterable(partitions_valid_imgs))
        all_valid_labels = list(itertools.chain.from_iterable(partitions_valid_lbls))
        pred = []
        y = []
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        writer = self.writer
        global_model = self.nn
        metric = 0
        global_model.eval()
        test_dicemetric = []

        dice_metric.reset()
        if self.options['modality'] =='CBF':
            max_intensity = 1200
        if self.options['modality'] =='CBV':
            max_intensity = 200
        if self.options['modality'] =='Tmax' or self.options['modality'] =='MTT':
            max_intensity = 30
        if self.options['modality'] =='ADC':
            max_intensity = 4000

        for path_test_case, path_test_label in zip(all_valid_paths,all_valid_labels):            
            test_vol = nib.load(path_test_case)
            test_lbl = nib.load(path_test_label)

            test_vol_pxls = test_vol.get_fdata()
            test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
            test_lbl_pxls = test_lbl.get_fdata()
            test_lbl_pxls = np.array(test_lbl_pxls)
            test_vol_pxls = (test_vol_pxls - 0) / (max_intensity - 0) 
            
            dices_volume =[]
            with torch.no_grad():
                for slice_selected in range(test_vol_pxls.shape[-1]):
                    out_test = global_model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
                    out_test = out_test.detach().cpu().numpy()
                    pred = np.array(out_test[0,0,:,:]>0.9, dtype='uint8')
                    cur_dice_metric = dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))                
                test_dicemetric.append(dice_metric.aggregate().item())
            # reset the status for next computation round
            dice_metric.reset()
        metric = np.mean(test_dicemetric)
        metric_values.append(metric)             
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = cur_epoch
            torch.save(global_model.state_dict(), self.options['modality']+'_'+self.options['suffix']+'_best_metric_model_segmentation2d_array.pth')
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                cur_epoch +1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_mean_dice", metric, cur_epoch)
        return best_metric, best_metric_epoch

    def global_test_cycle(self):                    
        partitions_test_imgs = [self.options['partitions_paths'][i][0][2] for i in range(len(self.options['partitions_paths']))]
        partitions_test_lbls = [self.options['partitions_paths'][i][1][2] for i in range(len(self.options['partitions_paths']))]
        all_test_paths  = list(itertools.chain.from_iterable(partitions_test_imgs))
        all_test_labels = list(itertools.chain.from_iterable(partitions_test_lbls))
        

        print("Loading best validation model weights: ")
        model_path = self.options['modality']+'_'+self.options['suffix']+'_best_metric_model_segmentation2d_array.pth'
        print(model_path)
        checkpoint = torch.load(model_path)
        self.nn.load_state_dict(checkpoint)
        model = self.nn

        pred = []
        test_dicemetric = []
        y = []
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        dice_metric.reset()
        if self.options['modality'] =='CBF':
            max_intensity = 1200
        if self.options['modality'] =='CBV':
            max_intensity = 200
        if self.options['modality'] =='Tmax' or self.options['modality'] =='MTT':
            max_intensity = 30
        if self.options['modality'] =='ADC':
            max_intensity = 4000

        for path_test_case, path_test_label in zip(all_test_paths,all_test_labels):            
            test_vol = nib.load(path_test_case)
            test_lbl = nib.load(path_test_label)

            test_vol_pxls = test_vol.get_fdata()
            test_vol_pxls = np.array(test_vol_pxls, dtype = np.float32)
            test_lbl_pxls = test_lbl.get_fdata()
            test_lbl_pxls = np.array(test_lbl_pxls)

            test_vol_pxls = (test_vol_pxls - 0) / (max_intensity - 0)

            for slice_selected in range(test_vol_pxls.shape[-1]):
                out_test = model(torch.tensor(test_vol_pxls[np.newaxis, np.newaxis, :,:,slice_selected]).to(device))
                out_test = out_test.detach().cpu().numpy()
                pred = np.array(out_test[0,0,:,:]>0.9, dtype='uint8')
                cur_dice_metric = dice_metric(torch.tensor(pred[np.newaxis,np.newaxis,:,:]),torch.tensor(test_lbl_pxls[np.newaxis,np.newaxis,:,:,slice_selected]))
            test_dicemetric.append(dice_metric.aggregate().item())
            # reset the status for next computation round
            dice_metric.reset()
        print("Global model test DICE for all slices: ")
        print(np.mean(test_dicemetric))
        return(np.mean(test_dicemetric))

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
                
        loss_function = monai.losses.DiceLoss(sigmoid=True, include_background=False)
        optimizer = Adam(ann.parameters(), local_lr)

        for epoch in range(local_epoch):
            for batch in dataloader_train:
                print(batch[0].shape)
                print(batch[1].shape)
            print("***")
            for batch_data in dataloader_train:
                inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
                print(inputs.shape, labels.shape)
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
        loss_function = monai.losses.DiceLoss(sigmoid=True,include_background=False)
        optimizer = ScaffoldOptimizer(ann.parameters(), lr=local_lr, weight_decay=1e-4)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
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
                
        loss_function = monai.losses.DiceLoss(sigmoid=True,include_background=False)

        optimizer = torch.optim.Adam(ann.parameters(), lr=local_lr)
        loss_generic = loss_function(torch.tensor(np.zeros((1,10))), torch.tensor(np.zeros((1,10))))
        loss_personalized = loss_function(torch.tensor(np.zeros((1,10))), torch.tensor(np.zeros((1,10))))
#        for k, v in self.nn.named_parameters():
#            print(v.requires_grad, v.dtype, v.device, k)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
                #(1)Optimization of the Generic path here equation (8) of the paper
                for k, v in ann.named_parameters(): #Transfering data from the generic head is done in dispatch()
                    v.requires_grad = True #deriving gradients to all the generic layers
                
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
               

                output_personalized = ann(inputs) + torch.tensor(output_generic).to(device) #regularized personalized output
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

class Centralized():
    def __init__(self):
        self.nn = UNet_custom(spatial_dims=2,
                            in_channels=1,
                            out_channels=1,
                            channels=(16, 32, 64, 128),
                            strides=(2, 2, 2),
                            kernel_size = (3,3),
                            num_res_units=2,
                            name='centralized').to(device)

    def train(self, ann, dataloader_train, local_epoch, local_lr):
        #First the generic encoder-decoder are updated       
        ann.train()
        ann.len = len(dataloader_train)
                
        loss_function = monai.losses.DiceLoss(sigmoid=True,include_background=False)

        optimizer = torch.optim.Adam(ann.parameters(), lr=local_lr)

        for epoch in range(local_epoch):
            for batch_data in dataloader_train:
                for k, v in ann.named_parameters():
                    v.requires_grad = True
                
                inputs, labels = batch_data[0][:,:,:,:,0].to(device), batch_data[1][:,:,:,:,0].to(device)
                y_pred_generic = ann(inputs)
                loss = loss_function(y_pred_generic, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return ann