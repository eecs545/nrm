import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

import tensorly as tl
tl.set_backend('pytorch')
from tensorly.tenalg import inner
from tensorly.random import check_random_state

def make_one_hot(labels, C=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    """
    target = th.eye(C)[labels.data]
    target = target.to(labels)      
    return target

#################### TRL class ############################################################
"""
https://github.com/JeanKossaifi/caltech-tutorial/blob/master/notebooks/6-tensor_regression_layer_pytorch.ipynb

Please refer to the above for the details. The source code of the TRL class is from the above link 
"""
class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)
            
        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)
            
        self.n_outputs = int(np.prod(output_size[1:]))
        
        # Core of the regression tensor weights
        
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])
        
        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        """Combine the core, factors and bias, with input x to produce the (1D) output
        """
        
        regression_weights = tl.tucker_to_tensor((self.core,self.factors))
        
        #return inner(x,regression_weights,tl.ndim(x)-1)+self.bias
        return regression_weights,(tl.tenalg.contract(x,[1,2,3],regression_weights,[0,1,2]) + self.bias)
    
    def penalty(self, order=1):
        """Add l2 regularization on the core and the factors"""
        penalty = tl.norm(self.core, order)
        
        for f in self.factors:
            penalty = penalty + tl.norm(f,order)
            
        return penalty 
    
#################### TRL class ############################################################


#################### Main NRM class ####################
class NRM(nn.Module):
    def __init__(self, batch_size, num_class):
        super(NRM, self).__init__()
        self.num_class = num_class
        self.batch_size = batch_size
        self.trl = TRL(ranks=(4, 3, 3, 3), input_size=(batch_size, 128, 8, 8), output_size=(batch_size,2))

        ######################################### FORWARD PASS LAYERS #############################################  
        
        conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn1 = nn.BatchNorm2d(32)
        lr1 = nn.LeakyReLU(0.1)
        mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn2 = nn.BatchNorm2d(64)
        lr2 = nn.LeakyReLU(0.1)
        mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        bn3 = nn.BatchNorm2d(128)
        lr3 = nn.LeakyReLU(0.1)
        mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                
        layers = [('conv0',conv1),('batchnorm0',bn1),('relu0',lr1),('pool1',mp1),('conv2',conv2),('batchnorm2',bn2),('relu2',lr2),('pool3',mp2),('conv4',conv3),('batchnorm4',bn3),('relu4',lr3),('pool5',mp3)]

        ######################################### FORWARD PASS LAYERS #############################################       

        
        ##############################################  NRM LAYERS #######################################################    
        
        nrm_bn1 = nn.BatchNorm2d(3)
        convtd1 = nn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        upsamp1 = nn.Upsample(scale_factor=2)
        
        convtd1.weight.data = conv1.weight.data
        
        nrm_bn2 = nn.BatchNorm2d(32)
        convtd2 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        upsamp2 = nn.Upsample(scale_factor=2)
        
        convtd2.weight.data = conv2.weight.data

        nrm_bn3 = nn.BatchNorm2d(64)
        convtd3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        upsamp3 = nn.Upsample(scale_factor=2)
        
        convtd3.weight.data = conv3.weight.data

        nrm_bn4 = nn.BatchNorm2d(128)        
        
        layers_nrm = [('batchnormtd0', nrm_bn1), ('convtd0',convtd1), ('upsample1',upsamp1), ('batchnorm1',nrm_bn2), ('convtd2', convtd2), ('upsample3', upsamp2), ('batchnorm3',nrm_bn3), ('convtd4',convtd3), ('upsample5', upsamp3), ('batchnorm5',nrm_bn4)]
        
        ################################################ NRM LAYERS #######################################################       
       
        ######################################### INSTANCE NORM NRM LAYERS ############################################
        
        insnrm_1 = nn.InstanceNorm2d(3)
        insnrm_2 = nn.InstanceNorm2d(32)
        insnrm_3 = nn.InstanceNorm2d(64)
        insnrm_4 = nn.InstanceNorm2d(128)
        
        insnorms_nrm = [insnrm_1,insnrm_2,insnrm_3,insnrm_4]
        
        ######################################### INSTANCE NORM NRM LAYERS ############################################

        
        ######################################### INSTANCE NORM CNN LAYERS ############################################
        
        inscnn_1 = nn.InstanceNorm2d(32)
        inscnn_2 = nn.InstanceNorm2d(64)
        inscnn_3 = nn.InstanceNorm2d(128)

        insnorms_cnn = [inscnn_1,inscnn_2,inscnn_3]
        
        ######################################### INSTANCE NORM CNN LAYERS ############################################

        model = nn.Sequential(OrderedDict(layers))
        
        self.layers = layers
        self.features = model 
        #layers_nrm = [nrm_bn1,convtd1, upsamp1,nrm_bn2,convtd2,upsamp2,nrm_bn3,convtd3,upsamp3,nrm_bn4]
       
        self.layers_nrm = layers_nrm[::-1]
        
        self.nrm = nn.Sequential(OrderedDict(layers_nrm[::-1]))
        self.insnorms_nrm = insnorms_nrm[::-1]
        self.insnorms_cnn = insnorms_cnn


    def forward(self, x, y=None):
        relu_latent = []; 
        pool_latent = [];
        bias_latent_cnn = []; 
        relu_latentpn = [];
        mean_latent_cnn = []; 
        var_latent_cnn = []
        xbias = th.zeros([1, x.shape[1], x.shape[2], x.shape[3]], device=None)              
        
        ############################  conv1 #####################################
        x = self.features[0](x)
        xbias = self.features[0](xbias)
        mean_latent_cnn.append(th.mean(x, dim=(0,2,3), keepdim=True))
        var_latent_cnn.append(th.mean((x - th.mean(x, dim=(0,2,3), keepdim=True))**2, dim=(0,2,3), keepdim=True))
        
        
        ############################   batchnorm1 ##################################
        x = self.features[1](x)
        xbias = self.insnorms_cnn[0](xbias)
        bias_latent_cnn.append(self.features[1].bias) 
        ############################   relu1 ##################################
        x = self.features[2](x)
        xbias = self.features[2](xbias)
        relu_latent.append(th.gt(x,0).float() + th.le(x,0).float()*0.1)

        #relu_latent and relu_latentpn keeps track of the pixels pool_latent are activated in the leaky relu
        relu_latentpn.append(th.gt(xbias,0).float() + th.le(xbias,0).float()*0.1)

        ############################   pool1 ##################################

        pool_latent.append(th.ge(x-F.interpolate(self.features[3](x), scale_factor=2, mode='nearest'),0))
        #pool_latent records the locations where the original pixel values are greater than the ones after interpolation
        #from a max pooled output.
        x = self.features[3](x) #perform maxpooling on input image/activation
        xbias = self.features[3](xbias) #perform maxpooling on input bias/bias activation

        ############################  conv2 #####################################
        x = self.features[4](x)
        xbias = self.features[4](xbias)
        mean_latent_cnn.append(th.mean(x, dim=(0,2,3), keepdim=True))
        var_latent_cnn.append(th.mean((x - th.mean(x, dim=(0,2,3), keepdim=True))**2, dim=(0,2,3), keepdim=True))
        
        
        ############################   batchnorm2 ##################################
        x = self.features[5](x)
        xbias = self.insnorms_cnn[1](xbias)
        bias_latent_cnn.append(self.features[5].bias) 

        ############################   relu2 ##################################
        x = self.features[6](x)
        xbias = self.features[6](xbias)
        relu_latent.append(th.gt(x,0).float() + th.le(x,0).float()*0.1)

        #relu_latent and relu_latentpn keeps track of the pixels pool_latent are activated in the leaky relu
        relu_latentpn.append(th.gt(xbias,0).float() + th.le(xbias,0).float()*0.1)

        ############################   pool2 ##################################

        pool_latent.append(th.ge(x-F.interpolate(self.features[7](x), scale_factor=2, mode='nearest'),0))
        #pool_latent records the locations where the original pixel values are greater than the ones after interpolation
        #from a max pooled output.
        x = self.features[7](x) #perform maxpooling on input image/activation
        xbias = self.features[7](xbias) #perform maxpooling on input bias/bias activation

        ############################  conv3 #####################################
        x = self.features[8](x)
        xbias = self.features[8](xbias)
        mean_latent_cnn.append(th.mean(x, dim=(0,2,3), keepdim=True))
        var_latent_cnn.append(th.mean((x - th.mean(x, dim=(0,2,3), keepdim=True))**2, dim=(0,2,3), keepdim=True))
        
        
        ############################   batchnorm3 ##################################
        x = self.features[9](x)
        xbias = self.insnorms_cnn[2](xbias)
        bias_latent_cnn.append(self.features[9].bias) 

        ############################   relu3 ##################################
        x = self.features[10](x)
        xbias = self.features[10](xbias)
        relu_latent.append(th.gt(x,0).float() + th.le(x,0).float()*0.1)

        #relu_latent and relu_latentpn keeps track of the pixels pool_latent are activated in the leaky relu
        relu_latentpn.append(th.gt(xbias,0).float() + th.le(xbias,0).float()*0.1)

        ############################   pool3 ##################################

        pool_latent.append(th.ge(x-F.interpolate(self.features[11](x), scale_factor=2, mode='nearest'),0))
        #pool_latent records the locations where the original pixel values are greater than the ones after interpolation
        #from a max pooled output.
        x = self.features[11](x) #perform maxpooling on input image/activation
        xbias = self.features[11](xbias) #perform maxpooling on input bias/bias activation
        
        relu_latent = relu_latent[::-1]
        pool_latent = pool_latent[::-1]
        bias_latent_cnn = bias_latent_cnn[::-1]
        self.bias_latent_cnn = bias_latent_cnn
        relu_latentpn = relu_latentpn[::-1]
        mean_latent_cnn = mean_latent_cnn[::-1]
        var_latent_cnn = var_latent_cnn[::-1]

        
        # send the features into the classifier
        trl_w,z = self.trl(x)
        w_t = trl_w.permute(dims=(3,0,1,2))
        
        # do reconstruction via nrm
        # xhat: the reconstruction image
        # loss_pn: path normalization loss
        # use z to reconstruct instead of argmax z
        
        
        xhat, _, loss_pn, loss_neg = self.topdown(self.nrm,make_one_hot(y, self.num_class), relu_latent, pool_latent, bias_latent_cnn, tl.ones([1, z.size()[1]], device=None), relu_latentpn, mean_latent_cnn, var_latent_cnn,w_t) if y is not None else  self.topdown(self.nrm,make_one_hot(th.argmax(z.detach(), dim=1), self.num_class), relu_latent, pool_latent, bias_latent_cnn, tl.ones([1, z.size()[1]], device=None), relu_latentpn, mean_latent_cnn, var_latent_cnn,w_t)


        return [z, xhat, loss_pn, loss_neg]

    def topdown(self, net,xhat, relu_latent, pool_latent, bias_latent_cnn, xpn, relu_latentpn, mean_latent_cnn, var_latent_cnn,tensor_wts):
        

        mu = xhat
        mupn = xpn
        
        loss_pn = th.zeros([self.batch_size,], device=None)
        loss_neg = th.zeros([self.batch_size,], device=None)

        relu_latent_indx = 0; pool_latent_indx = 0; meanvar_indx = 0; insnormtd_indx = 0
        prev_name = ''
        
        mu = tl.tenalg.inner(mu.to(th.float32),tensor_wts,tl.ndim(mu)-1)

        mupn = tl.tenalg.inner(mupn,tensor_wts,tl.ndim(mu)-3)
        
        
        ##################################### BATCH NORM 4#######################################################
        mu = self.nrm[0](mu.float())
        mupn = self.insnorms_nrm[0](mupn)
        #####################################  UPSAMPLE 3 #######################################################
        mu = self.nrm[1](mu.float())
        mupn = self.nrm[1](mupn)
        #####################################  convtd 3 #######################################################
        mu = mu * relu_latent[0].type(th.FloatTensor).to(mu) 
        mupn = mupn * relu_latentpn[0].type(th.FloatTensor).to(mu)
        
        mu_b =  bias_latent_cnn[0].data.reshape((1, -1, 1, 1)) *mu
        mupn_b = bias_latent_cnn[0].data.reshape((1, -1, 1, 1)) * mupn

        #Use mu_b and mupn_b to find path normalization loss
        loss_pn_layer = th.mean(th.abs(mu_b - mupn_b), dim=(1,2,3))                    
        loss_pn = loss_pn + loss_pn_layer
       
        mu = mu * pool_latent[0].type(th.FloatTensor).to(mu) 
        mupn = mupn * pool_latent[0].type(th.FloatTensor).to(mu)
        
        mu = self.nrm[2](mu.float())
        mupn = self.nrm[2](mupn)
        relu = nn.ReLU(inplace=True)
        loss_neg += th.norm(relu(-mu.view(mu.shape[0],-1)),dim=1)

        
        ##################################### BATCH NORM 3#######################################################
        mu = self.nrm[3](mu.float())
        mupn = self.insnorms_nrm[1](mupn)
        
        #####################################  UPSAMPLE 2 #######################################################
        mu = self.nrm[4](mu.float())
        mupn = self.nrm[4](mupn)
        
        #####################################  convtd 2 #######################################################
        mu = mu * relu_latent[1].type(th.FloatTensor).to(mu) 
        mupn = mupn * relu_latentpn[1].type(th.FloatTensor).to(mu)

        mu_b = mu * bias_latent_cnn[1].data.reshape((1, -1, 1, 1)) 
        mupn_b = mupn * bias_latent_cnn[1].data.reshape((1, -1, 1, 1)) * mupn

        #Use mu_b and mupn_b to find path normalization loss
        loss_pn_layer = th.mean(th.abs(mu_b - mupn_b), dim=(1,2,3))                    
        loss_pn = loss_pn + loss_pn_layer
       
        mu = mu * pool_latent[1].type(th.FloatTensor).to(mu) 
        mupn = mupn * pool_latent[1].type(th.FloatTensor).to(mu)
        
        mu = self.nrm[5](mu.float())
        mupn = self.nrm[5](mupn)
        relu = nn.ReLU(inplace=True)
        loss_neg += th.norm(relu(-mu.view(mu.shape[0],-1)),dim=1)

        ##################################### BATCH NORM 2#######################################################
        mu = self.nrm[6](mu.float())
        mupn = self.insnorms_nrm[2](mupn)
        
        #####################################  UPSAMPLE 1 #######################################################
        mu = self.nrm[7](mu.float())
        mupn = self.nrm[7](mupn)
        
        #####################################  convtd 1 #######################################################
        mu = mu * relu_latent[2].type(th.FloatTensor).to(mu) 
        mupn = mupn * relu_latentpn[2].type(th.FloatTensor).to(mu)
        
        mu_b = mu * bias_latent_cnn[2].data.reshape((1, -1, 1, 1)) 
        mupn_b = mupn * bias_latent_cnn[2].data.reshape((1, -1, 1, 1)) * mupn

        #Use mu_b and mupn_b to find path normalization loss
        loss_pn_layer = th.mean(th.abs(mu_b - mupn_b), dim=(1,2,3))                    
        loss_pn = loss_pn + loss_pn_layer
       
        mu = mu * pool_latent[2].type(th.FloatTensor).to(mu) 
        mupn = mupn * pool_latent[2].type(th.FloatTensor).to(mu)
        
        mu = self.nrm[8](mu.float())
        mupn = self.nrm[8](mupn)
        
        ##################################### BATCH NORM 1 #######################################################
        mu = self.nrm[9](mu.float())
        mupn = self.nrm[9](mupn)
       
        return mu, mupn, loss_pn, loss_neg