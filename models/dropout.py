import torch
import torch.nn as nn
import math
import numpy as np




class Pdropout(nn.Module):
    def __init__(self,p=0,ic=1):
        super(Pdropout,self).__init__()
        if not(0 <= p <= 1):
            raise ValueError("Drop rate must be in range [0,1]")
        self.p = p 
    def forward(self,input):
        if not self.training:
            return input
        else:
            importances = torch.mean(input,dim=1,keepdim=True)
            importances = torch.sigmoid(importances)
            #print(importances)
            mask = self.generate_mask(importances,input)
            
            #print(mask)
            input = input*mask
            return input
        
    def generate_mask(self,importance,input):
        n,f = input.shape
        #interpolation = torch.linspace(0,self.p,steps=n).view(-1,1).to(input.device)
        interpolation = self.non_linear_interpolation(self.p,0,n).to(input.device)
        #print(interpolation)
        mask = torch.zeros_like(importance)
        mask = mask.to(input.device)
        _, indx = torch.sort(importance,dim=0)
        #print(indx)
        idx = indx.view(-1)
        mask.index_add_(0,idx,interpolation)
        #print(mask)
        
        #mask 
        sampler = torch.rand(mask.shape[0],mask.shape[1]).to(input.device)
        #sampler = torch.rand_like(mask.shape[0],mask.shape[1])
        #mask = torch.bernoulli(mask)
        mask = (sampler < mask).float()
        mask = 1 - mask
        return mask
    
    def non_linear_interpolation(self,max,min,num):
        e_base = 20
        log_e = 1.5
        res = (max - min)/log_e* np.log10((np.linspace(0, np.power(10,(log_e)) - 1, num)+ 1)) + min
        #res = (max-min)/e_base *(np.power(10,(np.linspace(0, np.log10(e_base+1), num))) - 1) + min
        #res = (max - min)*(0.5*(1-np.cos(np.linspace(0, math.pi, num)))) + min
        res = torch.from_numpy(res).float()
        return res

class LinearScheduler(nn.Module):
    def __init__(self, model, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.model = model
        self.i = 0
        self.dropoutLayers = []
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))
        self.drop_values = self.dropvalue_sampler(start_value,stop_value,int(nr_steps))
        for name, layer in model.named_modules():
            if isinstance(layer, Pdropout):
                #print(name, layer)
                self.dropoutLayers.append(layer)
    def step(self):
        #for name, layer in model.named_modules():

        #dropout = []
        if self.i < len(self.drop_values):
            for i in range(len(self.dropoutLayers)):
                self.dropoutLayers[i].p = self.drop_values[self.i]
                # print(self.dropoutLayers[i].p)

        self.i += 1

    def dropvalue_sampler(self,min,max,num):
        e_base = 20
        log_e = 1.5
        res = (max - min)/log_e* np.log10((np.linspace(0, np.power(10,(log_e)) - 1, num)+ 1)) + min
        #res =  (max - min)*(0.5*(1-np.cos(np.linspace(0, math.pi, num)))) + min
        #res = (max-min)/e_base *(np.power(10,(np.linspace(0, np.log10(e_base+1), num))) - 1) + min

        return res
        