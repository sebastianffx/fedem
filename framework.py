import copy
import torch
import random
import monai
import numpy as np

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

    def train_server(self, global_epoch, local_epoch, global_lr, local_lr):
        for cur_epoch in range(global_epoch):
            print("*** global_epoch:", cur_epoch+1, "***")

            #skiping center 2 as only 1 scan is available (train_loader sorted)
            index=[0,1,2]
            #perform sampling here if desired

            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index, local_epoch, local_lr, cur_epoch)
            # aggregation
            self.aggregation(index, global_lr)

        return self.nn

    def client_update(self, index, local_epoch, local_lr, cur_epoch):
        tmp=0
        for i in index:
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

    def test(self, dataloader_test, test=True):
        model=self.nn
        model.eval()
        pred = []
        y = []
        dice_metric.reset()
        for test_data in dataloader_test:
            with torch.no_grad():
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
            for batch_data in dataloader_train:
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