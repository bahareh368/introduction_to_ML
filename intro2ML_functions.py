# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from autograd import grad 
# import autograd-wrapped numpy
import autograd.numpy as np  # install: pip install autograd
#%matplotlib inline
import matplotlib.pyplot as plt
#import numpy as np


def gradient_descent(g,alpha,max_its,w):
    
    # compute gradient module using autograd
    gradient = grad(g)

    # gradient descent loop
    weight_history = [w]              # weight history container
    cost_history = [g(w)]             # cost function history container
    for k in range(max_its):
        
        # evaluate the gradient
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
        
    return weight_history,cost_history


# standard normalization function - returns functions for standard normalizing and reverse standard
# normalizing an input dataset x
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds

    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means

    # return normalizer 
    return normalizer,inverse_normalizer








