#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


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


# In[ ]:





# In[6]:


# prepare data loaders
channel_stats = dict(mean=[0.5, 0.5, 0.5],
                        std=[0.5,  0.5,  0.5])

#For training data we perform Vertical and Horizontal Flips along with normalization as a part of data augmentation. 
#For validation and testing data we only do normalization.
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

    # load the dataset
    data_path = '../pipeline/cutout/nrm_training/'


    train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=train_transform)


    valid_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=valid_transform)


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
                           augment=False,
                           random_seed=2,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False)


robopol_artifact_loader = th.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='../pipeline/cutout/test/stars/',transform=eval_transformation),
    batch_size=4,
    shuffle=False,
    num_workers=2 * opt.workers,  # Needs images twice as fast
    pin_memory=False,
    drop_last=False)

ood_test_data_loader = th.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='../pipeline/cutout/test/nrm_test/',transform=eval_transformation),
    batch_size=4,
    shuffle=False,
    num_workers=2 * opt.workers,  # Needs images twice as fast
    pin_memory=False,
    drop_last=False)


# In[3]:


def get_loglikelihood(net, data_loader):
    neg_log = np.array([])
    net.eval()
    for i, (batch, target) in enumerate(data_loader):

        with th.no_grad():
            input_var = th.autograd.Variable(batch).to(device=None)

            [_, _, loglikeli, _] = net(input_var)
            neg_log = np.append(neg_log, loglikeli.cpu().detach().numpy()) 
            
    return -neg_log


# In[ ]:


from nrm_nonneg import NRM
model_gauss = NRM(batch_size=batch_size, num_class=2).to(device)
model_gauss.load_state_dict(th.load("models/model1.pth"),strict=False)

train_pn = get_rpn(model_gauss, robopol_artifact_loader, len(robopol_artifact_loader))
ood_pn = get_rpn(model_gauss, ood_test_data_loader, len(ood_test_data_loader))

volume = 64*64*3
rpn_losses = [train_pn*volume, ood_pn*volume]
colors = ["blue", "lime"]
dataset_name = ["Training Stars", "Galaxies"]



plt.rcParams.update({'font.size': 10})

for i in range(len(rpn_losses)):
    plt.hist(rpn_losses[i], bins="auto", normed=True, alpha=0.5, label = dataset_name[i], color = colors[i])
plt.legend(loc='upper left', prop={'size': 16})
plt.xlabel("Log joint likelihood of latent variables")


# In[7]:


labels_val = np.array([])
yhat = np.array([])
probs = np.array([])
for batch,target in ood_test_data_loader:
    
    
    with th.no_grad():
        input_var = th.autograd.Variable(batch)

        [output, _, _, _] = model_gauss(input_var)
        probability_artifact = th.nn.Softmax()(output)
        probs=np.append(probs,probability_artifact[:,1])
        
        
        pred = th.argmax(output, dim=1, keepdim=False).cpu().numpy()
        
        labels_val = np.append(labels_val,target)        
        yhat = np.append(yhat, pred)


# In[ ]:


from sklearn.metrics import roc_curve,f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score,auc,matthews_corrcoef,confusion_matrix
import scikitplot as skplt



conf=confusion_matrix(labels_val,yhat) 
print(conf) #Print confusion Matrix
print("F1 Score is : ",f1_score(labels_val,yhat))
print("Matthews Correlation Coefficient",matthews_corrcoef(labels_val,yhat))
print("Recall is : ",recall_score(labels_val,yhat))
print("Precision score is : ",precision_score(labels_val,yhat))

roc_sc=roc_auc_score(labels_val,probs) #Finding Area under curve of ROC
print("Area under curve is ",roc_sc)


plt.rcParams["figure.figsize"] = [11,11]
plt.rcParams["axes.labelsize"] = 24
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 25
fpr, tpr, thresholds = roc_curve(labels_val,probs)
plt.plot(fpr,tpr,'b')
plt.plot(fpr,fpr,'r')
s=900
plt.scatter(0.05624,0.9346,label='Threshold: 0.2',marker='.',s=s)
plt.scatter(0.03125,0.9797,label='Threshold: 0.5',marker=',',s=s)
plt.scatter(0.03125,0.98,label='Threshold: 0.75',marker='^',s=s)
plt.scatter(0.025,0.9595,label='Threshold: 0.9',marker='v',s=s)
plt.scatter(0,0.8433,label='Threshold: 0.99',marker='o',s=s)
plt.ylim(0.8,1)
plt.xlim(-0.001,0.17)

plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.legend(['AUC=0.996','AUC=0.500'])
#plt.savefig('ROC_cuve_zoom_jpg.png',dpi=150)

print(np.array(thresholds,dtype='float16'))
print(np.array(fpr,dtype='float16'))
print(np.array(tpr,dtype='float16'))


# In[ ]:


plt.rcParams["axes.labelsize"]=30
plt.rcParams["xtick.labelsize"]=30
plt.rcParams["ytick.labelsize"]=30
roc_auc = auc(fpr, tpr) # compute area under the curve
plt.plot(fpr, tpr,'b-',label='AUC=0.996',)
plt.plot([0, 1], [0, 1], 'r:',label='AUC=0.500')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
plt.legend(loc="best")
plt.show()
#plt.savefig('rocnew.jpg',dpi=100)

plt.close()

