import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,f1_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

from models.classic_attmil import AttnMIL,GatedAttention
from models.dropout import LinearScheduler

from utils import *
from wsi_dataloader_3 import C16DatasetV3,C16DatasetV3_tcga
from scipy import linalg
from timm.optim.adamp import AdamP
from radam import RAdam
from lookhead import Lookahead

def train(trainloader, milnet, criterion, optimizer, args):
    milnet.train()
    total_loss = 0
    bc = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.cuda()
        bag_label = label.cuda()
        bag_feats = bag_feats.view(-1, args.feats_size)

        optimizer.zero_grad()
        bag_prediction, _ = milnet(bag_feats)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        loss = bag_loss 
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(milnet.parameters(),5.0)
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (batch_id, len(trainloader), loss.item()))

    return total_loss / len(trainloader)

def test(testloader, milnet, criterion, args):
    milnet.eval()
    # csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)

            bag_prediction, _ = milnet(bag_feats)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            loss = bag_loss 
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            test_labels.extend([label.squeeze().cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
            
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
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
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--dataroot', default="dataset/tcga-lung/feats/ImageNet", type=str, help='dataroot for the CAMELYON16 dataset')
    parser.add_argument('--backgrd_thres', default=30, type=int, help='background threshold')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--drop_p', default=0.0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--model', default='attn', type=str, help='MIL model [dsmil]')
    parser.add_argument('--save_dir', default='abmil_imagenetresnet_c16', type=str, help='the directory used to save all the output')
    parser.add_argument('--fold', default='0', type=str, help='fold')
    args = parser.parse_args()
    #gpu_ids = tuple(args.gpu_index)
    #os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_index)
    torch.cuda.set_device(args.gpu_index)
    maybe_mkdir_p(join(args.save_dir, args.model))
    args.save_dir = make_dirs(join(args.save_dir, args.model))
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
    if args.model == "attn":
        milnet = AttnMIL(args.feats_size,args.num_classes,args.drop_p).cuda()

    elif args.model == 'gate':
        milnet = GatedAttention(args.feats_size,args.num_classes,args.drop_p).cuda()


    #milnet = AttentionMIL(args.feats_size, args.num_classes, args.L, args.D, args.attn_mode, args.dropout_node).cuda()
    optimizer = torch.optim.AdamW(milnet.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    #optimizer = AdamP(milnet.parameters(), lr=args.lr, wd_ratio=0.01, nesterov=True, weight_decay=args.weight_decay)
    #optimizer = RAdam(milnet.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    #optimizer = Lookahead(optimizer)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 1e-5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100],gamma=0.2)
    
    trainset = C16DatasetV3(args, 'train')
    testset = C16DatasetV3(args, 'test')
    
    #trainset = C16DatasetV3_tcga(args, 'train')
    #testset = C16DatasetV3_tcga(args, 'test')

    trainloader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, 1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        # train_path = shuffle(train_path).reset_index(drop=True)
        # test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(trainloader, milnet, criterion, optimizer, args) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal,f1 = test(testloader, milnet, criterion, args)
        
        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, f1 score: %.4f, AUC: ' % 
            (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score,f1) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

        
        #scheduler.step()
        current_score = (sum(aucs) + avg_score)/2
        if current_score >= best_score:
            best_score = current_score
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
            
            logger.info('Best model saved at: ' + save_name)
            logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            

if __name__ == '__main__':
    main()