import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision

import os
import datetime
import time



def get_acc(output, label):
    pred = th.argmax(output, dim=1, keepdim=False)
    correct = th.mean((pred == label).type(th.FloatTensor))
    return correct

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        th.nn.init.kaiming_uniform_(m.weight)
#         th.nn.init.xavier_uniform(m.weight)
        # m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Bias') != -1:
        m.bias.data.fill_(0)

class Trainer:
    
    """Class for training model"""

    def __init__(self, opt, device, net, NO_LABEL=-1):
        self.device = device
        self.net = net
                
        self.criterion = nn.CrossEntropyLoss(reduce=False, ignore_index=NO_LABEL)

        self.num_epochs = opt.num_epochs
        self.lr = opt.lr
        self.wd = opt.weight_decay
        self.decay_val = None
        self.alpha_kl = opt.alpha_kl
        self.alpha_bnmm = opt.alpha_bnmm
        self.alpha_pn = opt.alpha_pn
        self.alpha_neg = opt.alpha_neg

        self.model_dir = opt.model_dir
        self.exp_name = opt.exp_name
        self.last_epoch_trained = 0
        
        self.batch_size = opt.batch_size
        self.labeled_batch_size = opt.labeled_batch_size

    def train(self, train_loader, eval_loader):
        
        training_accuracy = []
        validation_accuracy = []
        
        training_loss = []        
        train_reco_loss = []
        train_l2_loss = []
        train_pn_loss = []
         
        validation_loss = []
        val_reco_loss = []
        val_l2_loss = []
        val_pn_loss = []
        
        trainer = th.optim.Adam(self.net.parameters(), self.lr[0], weight_decay=self.wd)
        best_valid_acc = 0
        iter_indx = 0
        
        
        for epoch in range(self.last_epoch_trained, self.num_epochs):
            
            start = time.time()
            train_loss = 0
            train_loss_xentropy = 0 
            train_loss_reconst = 0 
            train_loss_pn = 0 
            train_loss_kl = 0
            train_loss_neg = 0
            correct = 0
            num_batch_train = 0
                
            learning_rate = self.lr[0]
         
            # switch to train mode
            self.net.train()
        
            end = time.time()
            for i, (batch, target) in enumerate(train_loader):
                            
                # set up unlabeled input and labeled input with the corresponding labels
                    
                input_sup_var = th.autograd.Variable(batch[(self.batch_size - self.labeled_batch_size):]).to(self.device)

                target_sup_var = th.autograd.Variable(target[(self.batch_size - self.labeled_batch_size):]).to(self.device)

                # compute loss for labeled input                  

                [output_sup, xhat_sup, loss_pn_sup, loss_neg_sup] = self.net(input_sup_var, target_sup_var)

                loss_xentropy_sup = self.criterion(output_sup, target_sup_var)

                loss_reconst_sup = 0
                softmax_sup = F.softmax(output_sup, dim=1)

                loss_kl_sup = -th.sum(th.log(10.0*softmax_sup + 1e-8) * softmax_sup, dim=1)

                loss_sup = loss_xentropy_sup + self.alpha_kl * loss_kl_sup + self.alpha_pn * loss_pn_sup 
                
                # compute loss
                loss = th.mean(loss_sup)
                
                # compute the grads and update the parameters
                trainer.zero_grad()
                loss.backward()
                trainer.step()
            
                # accumulate all the losses for visualization
                loss_pn = loss_pn_sup
                loss_xentropy = loss_xentropy_sup
                loss_kl = loss_kl_sup
                loss_neg = loss_neg_sup
                train_loss_xentropy += th.mean(loss_xentropy).item() 
            
                train_loss_pn += th.mean(loss_pn).item() 
                train_loss_kl += th.mean(loss_kl).item() 
                train_loss_neg += th.mean(loss_neg).item() 
                train_loss += th.mean(loss).item()
                correct += get_acc(output_sup, target_sup_var).item()
                
                num_batch_train += 1
                iter_indx += 1

            self.last_epoch_trained += 1
            
            print("Epoch number : ",epoch)
            print("Training loss : ",train_loss/num_batch_train)
            print("Training PN loss : ",train_loss_pn/num_batch_train)
            print("Training Cross Entropy loss : ",train_loss_xentropy / num_batch_train)
            print("Training Negative loss : ",train_loss_neg/num_batch_train)
            print("Reconstruction loss : ",train_loss_reconst / num_batch_train)
            print("Training Accuracy : ",correct/num_batch_train)
            
         

            # Validation
            valid_loss = 0;
            valid_loss_xentropy = 0; 
            valid_loss_reconst = 0; 
            valid_loss_pn = 0; 
            valid_loss_kl = 0; 
            valid_loss_neg = 0
            valid_correct = 0
            num_batch_valid = 0

            self.net.eval()

            for i, (batch, target) in enumerate(eval_loader):
                with th.no_grad():
                    input_var = th.autograd.Variable(batch).to(self.device)
                    target_var = th.autograd.Variable(target).to(self.device)
                    
                    [output, xhat, loss_pn, loss_neg] = self.net(input_var, target_var)
                

                    loss_xentropy = self.criterion(output, target_var)
                    
                    softmax_val = F.softmax(output, dim=1)
                    loss_kl = -th.sum(th.log(10.0*softmax_val + 1e-8) * softmax_val, dim=1)
                    loss = loss_xentropy + self.alpha_kl * loss_kl + self.alpha_pn*loss_pn
                    #loss = loss_xentropy + self.alpha_reconst * loss_reconst + self.alpha_kl * loss_kl + self.alpha_pn * loss_pn +self.alpha_neg * loss_neg
                    valid_loss_xentropy += th.mean(loss_xentropy).item() 
                    valid_loss_pn += th.mean(loss_pn).item() 
                    valid_loss_kl += th.mean(loss_kl).item() 
                    valid_loss_neg += th.mean(loss_neg).item() 
                    valid_loss += th.mean(loss).item() 
                    
                    pred_here = th.argmax(output, dim=1, keepdim=False)
                    valid_correct += get_acc(output, target_var).item() 
                    
                    
                    num_batch_valid += 1
            valid_acc = valid_correct / num_batch_valid
            validation_accuracy.append(valid_acc)
            validation_loss.append(valid_loss)
            val_reco_loss.append(valid_loss_reconst)
            val_l2_loss.append(valid_loss_neg)
            val_pn_loss.append(valid_loss_pn)
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                th.save(self.net.state_dict(), '%s/%s_best.pth'%(self.model_dir, self.exp_name))
                
                
            print("Epoch number : ",epoch)
            print("Validation loss : ",valid_loss/num_batch_valid)
            print("Validation Cross Entropy loss : ",valid_loss_xentropy / num_batch_valid)
            print("Validation Negative loss : ",valid_loss_neg/num_batch_valid)
            print("Validation Reconstruction loss : ",valid_loss_reconst / num_batch_valid)
            print("Validation Accuracy : ",valid_correct/num_batch_valid)
            print("Validation PN Loss : ",valid_loss_pn/num_batch_valid)
            
            training_accuracy.append(correct/num_batch_train)
            training_loss.append(train_loss)
            train_pn_loss.append(train_loss_pn)
            train_reco_loss.append(train_loss_reconst)
            train_l2_loss.append(train_loss_neg)
            
        
            end = time.time()
            
            print("Time for epoch ",epoch,": ",end-start,"seconds")
            print("\n")
        
        return best_valid_acc,training_accuracy,validation_accuracy,training_loss,validation_loss,train_reco_loss,train_l2_loss,val_reco_loss,val_l2_loss,train_pn_loss,val_pn_loss
    

    def run_train(self, train_loader, eval_loader, num_exp):
        
        for i in range(num_exp):
            
            acc,train_acc,val_acc,train_loss,val_loss,train_reco_loss,train_l2_loss,val_reco_loss,val_l2_loss,training_pn_loss,val_pn_loss =self.train(train_loader, eval_loader)
            
            with open('train_acc.txt','w') as f:
                
                for item in train_acc:
                    f.write("%s\n" % item)
                    
                
            with open('valid_acc.txt','w') as f:
                for item in val_acc:
                    f.write("%s\n" % item)
                    
            with open('train_loss.txt','w') as f:
                
                for item in train_loss:
                    f.write("%s\n" % item)
                    
                
            with open('valid_loss.txt','w') as f:
                for item in val_loss:
                    f.write("%s\n" % item)
                    
            with open('train_reco_loss.txt','w') as f:
                
                for item in train_reco_loss:
                    f.write("%s\n" % item)
                    
            with open('valid_reco_loss.txt','w') as f:
                for item in val_reco_loss:
                    f.write("%s\n" % item)   
                    
            with open('train_l2_loss.txt','w') as f:
                
                for item in train_l2_loss:
                    f.write("%s\n" % item)
            
            with open('train_pn_loss.txt','w') as f:
                for item in training_pn_loss:
                    f.write("%s\n" % item)
                    
            with open('valid_l2_loss.txt','w') as f:
                for item in val_l2_loss:
                    f.write("%s\n" % item)
                    
            with open('valid_pn_loss.txt','w') as f:
                for item in val_pn_loss:
                    f.write("%s\n" % item)
