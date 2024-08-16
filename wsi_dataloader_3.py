# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:34:22 2023

@author: Xiwen Chen
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:38:14 2023

@author: Xiwen Chen
"""

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import sys, argparse, os
from utils import *



# def reOrganize_mDATA_test(mDATA):

#     tumorSlides = os.listdir(testMask_dir)
#     tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

#     SlideNames = []
#     FeatList = []
#     Label = []
#     for slide_name in mDATA.keys():
#         SlideNames.append(slide_name)

#         if slide_name in tumorSlides:
#             label = 1
#         else:
#             label = 0
#         Label.append(label)

#         patch_data_list = mDATA[slide_name]
#         featGroup = []
#         for tpatch in patch_data_list:
#             tfeat = torch.from_numpy(tpatch['feature'])
#             featGroup.append(tfeat.unsqueeze(0))
#         featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
#         FeatList.append(featGroup)

#     return SlideNames, FeatList, Label



def reOrganize_mDATA_test(mDATA,args, decimals=6):

    # tumorSlides = os.listdir(testMask_dir)
    # tumorSlides = [sst.split('.')[0] for sst in tumorSlides]
    

    data_csv = pd.read_csv(join(args.dataroot, 'test_offical.csv'))

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        # if slide_name in tumorSlides:
        #     label = 1
        # else:
        #     label = 0
            
        label = data_csv.loc[data_csv['subject_id']==slide_name]['bag_label'].item()

            
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        # featGroup = torch.round(featGroup, decimals=decimals)
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label

def reOrganize_mDATA(mDATA, decimals=6):

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        #print(slide_name)
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        # featGroup = torch.round(featGroup, decimals=decimals)
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


# dir_train = 'mDATA_train.pkl'
# dir_test = 'mDATA_test.pkl'


# with open(dir_train, 'rb') as f:
#     mDATA_train = pickle.load(f)

#     mDATA_val = pickle.load(f)
# with open(dir_test, 'rb') as f:
#     mDATA_test = pickle.load(f)
    

# SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train)
# # SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val)
# SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_test(mDATA_test)




class C16DatasetV4(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.dir_train = join(args.dataroot, 'mDATA_train.pkl') # fake_ground train_offical

            with open(self.dir_train, 'rb') as f:
                mDATA_train = pickle.load(f)
            
            #print(mDATA_train)
            self.SlideNames, self.FeatList, self.Label = reOrganize_mDATA(mDATA_train)
            
        elif split == 'test': 
            self.dir_test = join(args.dataroot, 'mDATA_test.pkl') #test_offical fake_test
            
            with open(self.dir_test, 'rb') as f:
                mDATA_test = pickle.load(f)
            
            self.SlideNames, self.FeatList, self.Label = reOrganize_mDATA_test(mDATA_test,args)
        elif split =='val': 
            self.data_csv =join(args.dataroot, 'val_offical.csv')
            


    def __getitem__(self, idx):
         label, feats = self.Label[idx], self.FeatList[idx]
         
         label_f = np.zeros(self.args.num_classes)
         if self.args.num_classes==1:
             label_f[0] = label
         else:
             # if int(csv_file_df.iloc[1])<=(len(label)-1):
             #     label[int(csv_file_df.iloc[1])] = 1
             label_f[int(label)] = 1
         
         # if self.split == 'train' and self.args.dropout_patch > 0.0:
         #     #print('drop')
         #     feats = dropout_patches(feats, self.args.dropout_patch)
             
             
         filename = self.SlideNames[idx]
            
         return {'feat':feats,'label':label_f, 'name': filename}
         
    def __len__(self):
         return len(self.Label )


class C16DatasetV3_tcga_dtfd(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.featureList = []
        self.bag_all = []
        if split == 'train':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_training.csv'))
        elif split == 'test':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_testing.csv'))
        elif split =='val':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_val.csv'))
            # drop_idx = []
            # for i in range(len(self.data_csv)):
            #     if self.data_csv.iloc[i, 0] in ['test_114', 'test_124']:
            #         drop_idx.append(i)
            # self.data_csv.drop(drop_idx, axis=0, inplace=True)
            # self.data_csv = self.data_csv.reset_index(drop=True)

        if isdir(join(args.dataroot, 'single_b' + str(args.backgrd_thres))):
            func = lambda row: join(args.dataroot, 'single_b' + str(args.backgrd_thres), row + ".csv")
        else:
            func = lambda row: join(args.dataroot, 'feats', row + ".csv")

        self.data_csv['0'] = self.data_csv['0'].apply(func)
        self.dir_list = self.data_csv['0'].to_list()
        #print(self.bag_name)
        self.label_list = self.data_csv['1'].to_list()
        #print(self.label_list)
        for i in range(len(self.dir_list)):
            label,feature = self.get_bag_feats(self.dir_list[i],self.label_list[i])
            self.featureList.append(feature) 
            self.bag_all.append(label)



    def get_bag_feats(self, dir,label_bag):
        feats_csv_path = dir

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = label_bag
        else:
            if int(label_bag)<=(len(label)-1):
                label[int( label_bag)] = 1

        label = torch.tensor(label_bag)
        feats = torch.tensor(np.array(feats)).float()

        return label, feats
    
    def __getitem__(self, idx):
        #label, feats = self.get_bag_feats(self.data_csv.iloc[idx])
        label =  self.bag_all[idx]  
        feats =  self.featureList[idx]
        if self.split == 'train' and self.args.dropout_patch > 0.0:
            #print('drop')
            feats = dropout_patches(feats, self.args.dropout_patch)

        return feats, label
        
    def __len__(self):
        return len(self.data_csv)


class C16DatasetV3_tcga(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.featureList = []
        self.bag_all = []
        if split == 'train':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_training.csv'))
        elif split == 'test':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_testing.csv'))
        elif split =='val':
            self.data_csv = pd.read_csv(join(args.dataroot, f'fold{args.fold}_val.csv'))
            # drop_idx = []
            # for i in range(len(self.data_csv)):
            #     if self.data_csv.iloc[i, 0] in ['test_114', 'test_124']:
            #         drop_idx.append(i)
            # self.data_csv.drop(drop_idx, axis=0, inplace=True)
            # self.data_csv = self.data_csv.reset_index(drop=True)

        if isdir(join(args.dataroot, 'single_b' + str(args.backgrd_thres))):
            func = lambda row: join(args.dataroot, 'single_b' + str(args.backgrd_thres), row + ".csv")
        else:
            func = lambda row: join(args.dataroot, 'feats', row + ".csv")

        self.data_csv['0'] = self.data_csv['0'].apply(func)
        self.dir_list = self.data_csv['0'].to_list()
        #print(self.bag_name)
        self.label_list = self.data_csv['1'].to_list()
        #print(self.label_list)
        for i in range(len(self.dir_list)):
            label,feature = self.get_bag_feats(self.dir_list[i],self.label_list[i])
            self.featureList.append(feature) 
            self.bag_all.append(label)



    def get_bag_feats(self, dir,label_bag):
        feats_csv_path = dir

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = label_bag
        else:
            if int(label_bag)<=(len(label)-1):
                label[int( label_bag)] = 1

        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()

        return label, feats
    
    def __getitem__(self, idx):
        #label, feats = self.get_bag_feats(self.data_csv.iloc[idx])
        label =  self.bag_all[idx]  
        feats =  self.featureList[idx]
        if self.split == 'train' and self.args.dropout_patch > 0.0:
            #print('drop')
            feats = dropout_patches(feats, self.args.dropout_patch)

        return feats, label
        
    def __len__(self):
        return len(self.data_csv)
        


class C16DatasetV3_dtfd(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.featureList = []
        self.bag_all = []
        if split == 'train':
            self.data_csv = pd.read_csv(join(args.dataroot, 'train_offical.csv'))
        elif split == 'test':
            self.data_csv = pd.read_csv(join(args.dataroot, 'test_offical.csv'))
        elif split =='val':
            self.data_csv = pd.read_csv(join(args.dataroot, 'val_offical.csv'))
            # drop_idx = []
            # for i in range(len(self.data_csv)):
            #     if self.data_csv.iloc[i, 0] in ['test_114', 'test_124']:
            #         drop_idx.append(i)
            # self.data_csv.drop(drop_idx, axis=0, inplace=True)
            # self.data_csv = self.data_csv.reset_index(drop=True)

        if isdir(join(args.dataroot, 'single_b' + str(args.backgrd_thres))):
            func = lambda row: join(args.dataroot, 'single_b' + str(args.backgrd_thres), row + ".csv")
        else:
            func = lambda row: join(args.dataroot, 'feats', row + ".csv")

        self.data_csv['subject_id'] = self.data_csv['subject_id'].apply(func)
        self.dir_list = self.data_csv['subject_id'].to_list()
        #print(self.bag_name)
        self.label_list = self.data_csv['bag_label'].to_list()
        #print(self.label_list)
        for i in range(len(self.dir_list)):
            label,feature = self.get_bag_feats(self.dir_list[i],self.label_list[i])
            self.featureList.append(feature) 
            self.bag_all.append(label)



    def get_bag_feats(self, dir,label_bag):
        feats_csv_path = dir

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = label_bag
        else:
            if int(label_bag)<=(len(label)-1):
                label[int( label_bag)] = 1

        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()

        return label_bag, feats
    
    def __getitem__(self, idx):
        #label, feats = self.get_bag_feats(self.data_csv.iloc[idx])
        label =  self.bag_all[idx]  
        feats =  self.featureList[idx]
        if self.split == 'train' and self.args.dropout_patch > 0.0:
            #print('drop')
            feats = dropout_patches(feats, self.args.dropout_patch)

        return feats, label
        
    def __len__(self):
        return len(self.data_csv)






class C16DatasetV3(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()
        self.args = args
        self.split = split
        self.featureList = []
        self.bag_all = []
        if split == 'train':
            self.data_csv = pd.read_csv(join(args.dataroot, 'train_offical.csv'))
        elif split == 'test':
            self.data_csv = pd.read_csv(join(args.dataroot, 'test_offical.csv'))
        elif split =='val':
            self.data_csv = pd.read_csv(join(args.dataroot, 'val_offical.csv'))
            # drop_idx = []
            # for i in range(len(self.data_csv)):
            #     if self.data_csv.iloc[i, 0] in ['test_114', 'test_124']:
            #         drop_idx.append(i)
            # self.data_csv.drop(drop_idx, axis=0, inplace=True)
            # self.data_csv = self.data_csv.reset_index(drop=True)

        if isdir(join(args.dataroot, 'single_b' + str(args.backgrd_thres))):
            func = lambda row: join(args.dataroot, 'single_b' + str(args.backgrd_thres), row + ".csv")
        else:
            func = lambda row: join(args.dataroot, 'feats', row + ".csv")

        self.data_csv['subject_id'] = self.data_csv['subject_id'].apply(func)
        self.dir_list = self.data_csv['subject_id'].to_list()
        #print(self.bag_name)
        self.label_list = self.data_csv['bag_label'].to_list()
        #print(self.label_list)
        for i in range(len(self.dir_list)):
            label,feature = self.get_bag_feats(self.dir_list[i],self.label_list[i])
            self.featureList.append(feature) 
            self.bag_all.append(label)



    def get_bag_feats(self, dir,label_bag):
        feats_csv_path = dir

        df = pd.read_csv(feats_csv_path)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = label_bag
        else:
            if int(label_bag)<=(len(label)-1):
                label[int( label_bag)] = 1

        label = torch.tensor(np.array(label))
        feats = torch.tensor(np.array(feats)).float()

        return label, feats
    
    def __getitem__(self, idx):
        #label, feats = self.get_bag_feats(self.data_csv.iloc[idx])
        label =  self.bag_all[idx]  
        feats =  self.featureList[idx]
        if self.split == 'train' and self.args.dropout_patch > 0.0:
            #print('drop')
            feats = dropout_patches(feats, self.args.dropout_patch)

        return feats, label
        
    def __len__(self):
        return len(self.data_csv)

class C16DatasetV2(Dataset):
    def __init__(self, dataroot, split='train', split_ratio=0.9, num_classes=2, dropout_patch=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_patch = dropout_patch
        self.split = split

        if split in ['train', 'val']:
            with open(join(dataroot, 'mDATA_train.pkl'), 'rb') as f:
                mDATA = pickle.load(f)

            mDATA_train , mDATA_val  = [i.to_dict() for i in train_test_split(pd.Series(mDATA), train_size=split_ratio, random_state=42)]

            if split == 'train':
                mDATA = mDATA_train
            else:
                mDATA = mDATA_val

            SlideNames, FeatList, Label = self.reOrganize_mDATA(mDATA)
        elif split == 'test':
            with open(join(dataroot, 'mDATA_test.pkl'), 'rb') as f:
                mDATA = pickle.load(f)
            test_info = pd.read_csv(join(dataroot, 'reference.csv'))
            SlideNames, FeatList, Label = self.reOrganize_mDATA_test(mDATA, test_info)

        self.SlideNames = SlideNames
        self.FeatList = FeatList
        self.Label = Label

    def reOrganize_mDATA_test(self, mDATA, test_info):
        tumorSlides = []
        for i in range(len(test_info)):
            if test_info.iloc[i, 1] == 'Tumor':
                tumorSlides.append(test_info.iloc[i, 0])

        SlideNames = []
        FeatList = []
        Label = []
        for slide_name in mDATA.keys():
            SlideNames.append(slide_name)

            if slide_name in tumorSlides:
                label = 1
            else:
                label = 0
            Label.append(label)

            patch_data_list = mDATA[slide_name]
            featGroup = []
            for tpatch in patch_data_list:
                tfeat = torch.from_numpy(tpatch['feature'])
                featGroup.append(tfeat.unsqueeze(0))
            featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
            FeatList.append(featGroup)

        return SlideNames, FeatList, Label

    def reOrganize_mDATA(self, mDATA):
        SlideNames = []
        FeatList = []
        Label = []
        for slide_name in mDATA.keys():
            SlideNames.append(slide_name)

            if slide_name.startswith('tumor'):
                label = 1
            elif slide_name.startswith('normal'):
                label = 0
            else:
                raise RuntimeError('Undefined slide type')
            Label.append(label)

            patch_data_list = mDATA[slide_name]
            featGroup = []
            for tpatch in patch_data_list:
                tfeat = torch.from_numpy(tpatch['feature'])
                featGroup.append(tfeat.unsqueeze(0))
            featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
            FeatList.append(featGroup)

        return SlideNames, FeatList, Label

    def __len__(self):
        return len(self.SlideNames)

    def __getitem__(self, idx):
        slide_name = self.SlideNames[idx]
        if self.dropout_patch > 0.0 and self.split == 'train':
            bag_feat = dropout_patches(self.FeatList[idx], self.dropout_patch)

        bag_feat = shuffle(bag_feat)
        bag_label = int(self.Label[idx])

        label = np.zeros(self.num_classes)
        if self.num_classes==1:
            label[0] = bag_label
        else:
            if bag_label <= (len(label) - 1):
                label[bag_label] = 1
        
        return slide_name, bag_feat, label
    
def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

class C16DatasetV1(Dataset):
    def __init__(self, dataroot, split="train", level=1, onehot_label=True, dropout_patch_rate=0.0, seed=0):
        super().__init__()
        self.split = split
        self.onehot_label = onehot_label
        self.dropout_patch_rate = dropout_patch_rate
        self.seed = seed
        self.base_dir = join(dataroot, split, f"{10.0 * level:.1f}", "extracted_features")   
        self.csv = shuffle(pd.read_csv(join(dataroot, split + "_offical.csv")).sort_values(by=['subject_id']), random_state=seed)
        self.bag_labels = self.csv.iloc[:, -1].values.tolist()

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        subject_id = self.csv.iloc[idx, 0]
        bag_label = int(self.csv.iloc[idx, 1])

        if self.split == 'train':
            data_file = join(self.base_dir, "normal" if bag_label == 0 else "tumor", subject_id + ".npy")
        elif self.split == 'test':
            data_file = join(self.base_dir, subject_id + ".npy")

        embed_feats = np.load(data_file)

        if self.onehot_label:
            bag_label_new = np.zeros(2, dtype=np.int32)
            bag_label_new[bag_label] = 1
            bag_label = bag_label_new
        
        embed_feats = self.augment(embed_feats)

        if self.split == 'train':
            if self.dropout_patch_rate and self.dropout_patch_rate > 0.0:
                embed_feats = dropout_patches(embed_feats, self.dropout_patch_rate)

        return embed_feats, bag_label
    
    def augment(self, feats):
        np.random.shuffle(feats)
        return feats











if __name__ == "__main__":
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--dataroot', default='../../DTML_feats/', type=str, help='dataroot for the CAMELYON16 dataset')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--dropout_patch', default=0.4, type=float, help='Patch dropout rate [0]')
    args = parser.parse_args()
    dataset = C16DatasetV4(args, "test")
    dataloader = DataLoader(dataset, 1, True, drop_last=False)
    ct=0
    for bag in dataloader:
        feats = bag['feat']
        label = bag['label']
        bag_name = bag['name']
        
        ct = ct+1
        print(ct)
        print(feats.size())
        print(label)
        break