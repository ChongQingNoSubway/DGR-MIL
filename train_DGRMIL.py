# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:34:41 2023

@author: Xiwen Chen
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:25:56 2023

@author: Xiwen Chen
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score,f1_score

 
from utils import *

from wsi_dataloader_3 import C16DatasetV3,C16DatasetV4,dropout_patches,C16DatasetV3_tcga

from torch.cuda.amp import GradScaler, autocast
                                        
                                        
import torch.nn.functional as F

 

 
 
from lookhead import Lookahead
import warnings

from models.dgrmil import DGRMIL

 

# Suppress all warnings
warnings.filterwarnings("ignore")

 
from scheduler import LinearWarmupCosineAnnealingLR                                    
                                        
                                        
def train(trainloader, milnet, criterion, optimizer, epoch, start,args,scaler):
    milnet.train()
    total_loss = 0
    bc = 0
    #loss_l1 = nn.L1Loss()
    for batch_id, (feats,label) in enumerate(trainloader):
 
        # bag = bag[0]
        # feats = bag['feat']
        # label = bag['label']
        # bag_name = bag['name'][0]
        bag_feats = feats.cuda()
        bag_label = label.cuda()
        #bag_feats = bag_feats.view(-1, args.feats_size)
        #bag_feats = feats.cuda()
        #bag_label = label.cuda()
        
        
        # bag_label = bag_label.repeat(args.num_les, 1)
        
        # print(bag_label)
        # bag_feats = bag_feats.view(-1, args.feats_size)

        optimizer.zero_grad()
 
        
        with autocast():
            if torch.argmax(label)==0:
                bag_prediction, A,H,p_center,nc_center,lesion = milnet(bag_feats,bag_mode='normal')
             
            else:
                bag_prediction, A,H,p_center,nc_center,lesion= milnet(bag_feats,bag_mode='abnormal')

            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

            if epoch<args.epoch_des:
                loss = bag_loss 
                sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), bag_loss.item(),loss.item()))
            else:
                lesion_norm = lesion.squeeze(0)
                lesion_norm = torch.nn.functional.normalize(lesion_norm)
                div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(args.num_les).cuda())
                #print(div_loss)
                sim_loss = tripleloss(lesion,p_center,nc_center)
                loss = bag_loss  +   0.1*div_loss + 0.1*sim_loss  
                sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f   sim_loss: %.4f  div loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), bag_loss.item(),sim_loss.item(),div_loss.item(),loss.item()))
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # singie precious
        # loss.backward()
        # optimizer.step()


        total_loss = total_loss + bag_loss
 

    return total_loss / len(trainloader)

def similarityloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    totall_loss = 0

    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)

    cosin = nn.CosineSimilarity()

    loss_nc = cosin(golabal,nc_center).mean()
     
    loss_p = -cosin(golabal,p_center).mean()
    #print(loss_p)
    totall_loss = loss_nc + loss_p
    
    return totall_loss


def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)

    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    
    loss = triple_loss(golabal,p_center,nc_center)

    return loss

def test(testloader, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for batch_id, (feats,label) in enumerate(testloader):
            
            # (feats, label)
            # bag = bag[0]
            # feats = bag['feat']
            # label = bag['label']
            # bag_name = bag['name']
            
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            #bag_feats = bag_feats.view(1, -1, args.feats_size)
            # print(bag_label)
            # bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction, A, H = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1,-1),bag_label.view(1,-1))
            loss = bag_loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            test_labels.extend([label.squeeze().cpu().numpy()])
            # if args.average:
            #     test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            # else: 
            #     test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    # print(test_labels)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(testloader)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(testloader)
    f1 = f1_score(test_labels,test_predictions,average='macro')
    print(f1)
    return total_loss / len(testloader), avg_score, auc_value, thresholds_optimal,f1

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1: 
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by Resnet50')
    parser.add_argument('--dataroot', default="dataset/c16/vit", type=str, help='dataroot for the CAMELYON16 dataset')
    parser.add_argument('--backgrd_thres', default=30, type=int, help='background threshold')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='our', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0.1, type=float, help='Patch dropout rate [0] 0.4')
    parser.add_argument('--dropout_node', default=0.4, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--num_cluster', default=5, type=int, help='number of assumped clusters')
    parser.add_argument('--num_les', default=5, type=int, help='number of representation for normal/lesion')
    parser.add_argument('--num_normal', default=5, type=int, help='number of representation for normal/lesion')
    parser.add_argument('--weight_div', default=0.2, type=float, help='weight for block loss default 0.0001')
    parser.add_argument('--weight_des', default=0.01, type=float, help='weight for block loss default 0.0001')
    parser.add_argument('--temp', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--aggmode', default='mean', type=str, help='aggregation mode')
    parser.add_argument('--optimizer', default='adamw', type=str, help='aggregation mode')
    parser.add_argument('--epoch_contrastive', default=200, type=int, help='turn on contrastive learning')
    parser.add_argument('--epoch_step', default='[100]', type=str)
    parser.add_argument('--save_dir', default='./tcga_resnetimagenet/fold0', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on neg pos descrimination')
    parser.add_argument('--fold', default='0', type=str, help='fold')
    parser.add_argument('--L', default=512, type=int)
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    maybe_mkdir_p(join(args.save_dir, f'{args.model}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.model}'))
    maybe_mkdir_p(args.save_dir)



    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


    criterion = nn.BCEWithLogitsLoss()
    
    # <------------- define MIL network ------------->
    if args.model == 'our':
        milnet = DGRMIL(args.feats_size,L=args.L,n_lesion=args.num_les,dropout_node=args.dropout_node,dropout_patch = args.dropout_patch).cuda()

        
    optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=args.epoch_des,max_epochs=args.num_epochs,warmup_start_lr=0,eta_min=1e-5)
    
    # C16 DATASET
    trainset = C16DatasetV3(args, 'train')
    testset = C16DatasetV3(args, 'test')

    # TCGA DATASET
    #trainset = C16DatasetV3_tcga(args, 'train')
    #testset = C16DatasetV3_tcga(args, 'test')
    
    scaler = GradScaler()
    trainloader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
 
    for epoch in range(1, args.num_epochs + 1):
 
        start = False
        if best_score > 0.8:
            start = True
        else:
            start = False
        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch,start,args,scaler) # iterate all bags
        
        test_loss_bag, avg_score, aucs, thresholds_optimal,f1 = test(testloader, milnet, criterion, args)
        
        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, f1 score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score,f1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        
        scheduler.step()
        current_score = (sum(aucs) + avg_score)/3
        if current_score >= best_score:
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
             
            logger.info('Best model saved at: ' + save_name)
            logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            


if __name__ == '__main__':
    main()