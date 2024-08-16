# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 18:19:20 2023

@author: Xiwen Chen
"""
import numpy as np
import math

import torch

def rd(X,epsilon2=0.5): #calcualate the additive diveristy
    # X = np.concatenate((X_candidate,X_previous),axis=0)
    # X = X- np.mean(X,axis = 0)
    n,m = X.shape
    
    if n>m:
        # t = time.time()
        _,s,_ = np.linalg.svd(X.T)
        # print(time.time()-t)
    else:
        # t = time.time()
        _,s,_ = np.linalg.svd(X)
        # print(time.time()-t)
        
        
    rate = np.sum(np.log(1+s**2/n*m /epsilon2  ) )
        
    return rate
    


# X= torch.rand(10,50)
# rd_torch(X,epsilon2=0.5)

# rd(X.numpy(),epsilon2=0.5)


def rd_torch(X,epsilon2=0.5): #calcualate the additive diveristy
    # X = np.concatenate((X_candidate,X_previous),axis=0)
    X = X- torch.mean(X,dim = 0)
    n,m = X.shape
    
    if n>m:
        # t = time.time()
        _,s,_ = torch.linalg.svd(X.T)
        # print(time.time()-t)
    else:
        # t = time.time()
        _,s,_ = torch.linalg.svd(X)
        # print(time.time()-t)
        
        
    rate = torch.sum(torch.log(1+s**2/n*m /epsilon2  ) )
        
    return rate


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix)) #shape为(item_size,)
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items



def rd_add(X_candidate,X_previous,epsilon2=0.5): #calcualate the additive diveristy
    X = np.concatenate((X_candidate,X_previous),axis=0)
    n,m = X.shape
    X = X-np.mean(X,axis=0)
    if n>m:
        # t = time.time()
        _,s,_ = np.linalg.svd(X.T)
        # print(time.time()-t)
    else:
        # t = time.time()
        _,s,_ = np.linalg.svd(X)
        # print(time.time()-t)
        
        
    rate = np.sum(np.log(1+s**2/n*m /epsilon2  ) )
        
    return rate




# def rd_add(X_candidate,X_previous,epsilon2=0.5): #calcualate the additive diveristy
#     X = np.concatenate((X_candidate,X_previous),axis=0)
#     n,m = X.shape
    
#     if n>m:
#         # t = time.time()
#         _,s,_ = np.linalg.svd(X.T)
#         # print(time.time()-t)
#     else:
#         # t = time.time()
#         _,s,_ = np.linalg.svd(X)
#         # print(time.time()-t)
        
        
#     rate = np.sum(np.log(1+s**2/n*m /epsilon2  ) )
        
#     return rate
