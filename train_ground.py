#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import argparse
import math
import logging

import datetime
import time

import numpy as np
import operator as op
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

import torchvision.transforms as transforms

from train_script import *


# In[7]:


# set up parameters
class Options:
    def __init__(self):
        self.seed_val = 1 # random seed val
        self.batch_size =4 # batch size
        self.labeled_batch_size = 4 # actually no use in supervised/unsupervised learning, only used in semisupervised learning
        self.device = -1 # gpu id
        
        self.lr = [0.01, 0.05] # learning rate for adam and then initial learning rate for sgd respectively
        self.num_epochs = 15 # number of training epochs
        self.weight_decay = 5e-4 # weight decay
        
        self.mount_point = '../' # change this to your mount_point
        
        self.datadir = '../pipeline/cutout/nrm_training/'
        self.log_dir = os.path.join(self.mount_point,'logs') # log directory
        
        
        self.model_dir = os.path.join(self.mount_point,'models') # log 
        
        self.train_subdir = '../pipeline/cutout/nrm_training/'
        self.eval_subdir = '../pipeline/cutout/test/nrm_test/'
        self.num_classes = 2
        self.workers = 0
        

opt = Options()
device = th.device("cuda:%i"%opt.device if (opt.device >= 0) else "cpu")


# prepare data loaders

#The channel stats is a dictionary that defines the values of mean and variance for each input channel in the image. 

channel_stats = dict(mean=[0.5, 0.5, 0.5],
                        std=[0.5,  0.5,  0.5])

train_transformation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])
eval_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

"""
Source code for Training and Validation Data Loaders:

Reference: https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123

"""

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the training dataset
    data_path = '../pipeline/cutout/nrm_training/'

    train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=train_transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=valid_transform)

    #split the dataset into training and validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = th.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = th.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)

data_path = '../pipeline/cutout/nrm_training/'

batch_size = opt.batch_size

train_loader,val_loader=get_train_valid_loader(data_path,
                           batch_size,
                           augment=True,
                           random_seed=2,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False)


# set random seed for weight init, change it for different experiments
manualSeed = opt.seed_val
np.random.seed(manualSeed)
th.manual_seed(manualSeed)

# import the NRM
from nrm_nonneg import NRM
model = NRM(batch_size=opt.batch_size, num_class=2).to(device)
model.apply(weights_init)
# train Gaussian model

model_train = Trainer(opt, device, model)

model_train.run_train(train_loader, val_loader, 1)



