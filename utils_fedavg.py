from torch import nn, optim

def get_optimizer(optimizer_name, cur_lr, model, cur_momentum, weight_decay=1e-4):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                          lr= cur_lr,
                          momentum=cur_momentum)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                           lr=cur_lr,
                           weight_decay=weight_decay)
    return optimizer
