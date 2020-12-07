#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import random
from collections import namedtuple

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/leaf.data", header=None)

data = df_train.iloc[:,1:]
labels = df_train.iloc[:,0]

def choose_centers_kpp(data, k):
    mu = [data.sample(1).values[0]]
    for i in range(k-1):
        find_dist_from_centers_kpp(data, mu)
        mu.append(choose_next_center(data))
    return mu

def find_dist_from_centers_kpp(data, mu):
    d2 = np.array([np.linalg.norm(x - mu[-1])**2 for x in data])
    global DX_2
    if len(DX_2) == 0:
        DX_2 = np.array(d2[:])
    else:
        for i in range(len(d2)):
            if d2[i] < DX_2[i]:
                DX_2[i] = d2[i]


def choose_next_center(data):
    global DX_2
    dx2_sum = sum(DX_2)
    probabilities = [x / dx2_sum for x in DX_2]
    cum_probabilities = np.cumsum(probabilities)
    
    rand = random.random()
    index = np.where(cum_probabilities >= rand)[0][0]
    return data.iloc[index,:]

    
def gmm(data, max_iters=10, k = 3, eps = 0.0001, plus_plus = False):
    num_data, num_features = data.shape
    
    if plus_plus == False:
        # random initialization of means
        mu = data.iloc[np.random.choice(num_data, k, False), :]
    else:
        temp = choose_centers_kpp(data, k)
        mu = pd.DataFrame(temp)
        indices=list(range(1, len(temp[0])+1))
        mu.columns = indices
        
    # initialize the covariance matrices
    sigma= [np.eye(num_features)] * k
    
    # initialize the probabilities for each gaussian
    w = [1.0/k] * k
    
    R = np.zeros((num_data, k))

    log_likelihoods = []

    while len(log_likelihoods) < max_iters:
    
        # E - Step
        for k in range(k):
            R[:, k] = w[k] * multivariate_normal(mu.iloc[k,:], sigma[k]).pdf(data)
            
            # Likelihood
            log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
            log_likelihoods.append(log_likelihood)
            
            # Normalize 
            R = (R.T / np.sum(R, axis = 1)).T
            
            # The number of datapoints belonging to each gaussian            
            N_ks = np.sum(R, axis = 0)
    
        # M Step
        for k in range(k):
            # means
            mu.iloc[k,:] = 1.0 / N_ks[k] * np.sum(R[:, k] * data.T, axis = 1).T
            x_mu = np.matrix(data - mu.iloc[k,:])
            
            # covariance
            sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
            
            # probabilities
            w[k] = 1.0/ num_data * N_ks[k]
            
        if len(log_likelihoods) < 2 : continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < eps: break
    
    params = namedtuple('params', ['mu', 'sigma', 'w', 'log_likelihoods'])
    params.mu = mu
    params.sigma = sigma
    params.w = w
    params.log_likelihoods = log_likelihoods
    
    return params

    
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

K = [12, 18, 24, 36, 42]
for k in K:
    print "k=",k
    params = gmm(data,100,k)
    print "Log likelihoods:"
    print params.log_likelihoods
    print "Mean:",sum(params.log_likelihoods)/20
    print "Variance:",np.var(params.log_likelihoods)
    
  
print "\n\nKmeans++ initialization GMM:"
for k in K:
    print "k=",k
    params = gmm(data,100,k, plus_plus=True)
    print "Log likelihoods:"
    print params.log_likelihoods
    print "Mean:",sum(params.log_likelihoods)/20
    print "Variance:",np.var(params.log_likelihoods)
