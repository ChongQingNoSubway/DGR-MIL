from infoNCE import InfoNCE

import torch 



def contrastiveloss(lesionrepresentation,postive_instances,negative_instances,temp):
    n_positive, _ = postive_instances.shape
    n_globallesionrepresente, _ = lesionrepresentation.shape
    totall_loss = 0
    for i in range(n_positive):
        #print(postive_instances[i,:].shape)
        n_postivelist = postive_instances[i,:].unsqueeze(0).repeat(n_globallesionrepresente, 1)
        loss = InfoNCE(temperature=temp,negative_mode='unpaired')
        loss = loss(lesionrepresentation, n_postivelist, negative_instances)
        totall_loss += loss
    return totall_loss/ n_positive


 
    