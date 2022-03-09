import copy
import torch
import numpy as np
from scipy.special import softmax


def average_weights_softmax(local_trained_weights, loaders_lengths):
    """Returns the average of the weights.
       local_trained_weights: The list of tensors (of the same shape) that will be averaged
    """
    weights_samples_scaling = softmax(loaders_lengths)    
    avg_weights = copy.deepcopy(local_trained_weights[0])
    
    for key in avg_weights.keys():
        avg_weights[key] = avg_weights[key]*weights_samples_scaling[0]
        
    for key in avg_weights.keys():
        for i in range(1, len(local_trained_weights)):
            avg_weights[key] += (weights_samples_scaling[i])*local_trained_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], np.array(weights_samples_scaling).sum())
    return avg_weights



def average_weights_beta(local_trained_weights, loaders_lengths, beta_val=0.9):

    weight_classes = [(1-beta_val)/(1-np.power(beta_val,length)) for length in loaders_lengths]

    """Returns the global weights using a beta-weighted average
       local_trained_weights: The list of tensors (of the same shape) that will be averaged
    """
    # Initialize copy model weights with the untrained model weights.
    avg_weights = copy.deepcopy(local_trained_weights[0])
    for key in avg_weights.keys():
        avg_weights[key] = avg_weights[key]*weight_classes[0]
        
    for key in avg_weights.keys():
        for i in range(1, len(local_trained_weights)):
            avg_weights[key] += (weight_classes[i])*local_trained_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], np.array(weight_classes).sum())
    return avg_weights




def average_weights(local_trained_weights):
    """Returns the average of the weights.
       local_trained_weights: The list of tensors (of the same shape) that will be averaged
    """
    # Initialize copy model weights with the untrained model weights.
    avg_weights = copy.deepcopy(local_trained_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_trained_weights)):
            avg_weights[key] += local_trained_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_trained_weights))
    return avg_weights
