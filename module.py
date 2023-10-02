import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
from operator import add
import time
import math
from utils import *
from copy import deepcopy as cp



class Module(object):
    def __init__(self, model, config, optimizer, criterion):
        
        self.model = model
            
        self.optimizer = optimizer
        self.epoch = -1
        if config['multistep']:
            self.scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config['milestone'], gamma=config['decay_rate'])
        elif config['lr_step'] > 0:
            self.scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['decay_rate'])
        elif config['cos_restart']:
            print('Using CosineAnnealingLR_Restart Scheduler!')
            self.scheduler = CosineAnnealingLR_Restart(optimizer,T_period=list(config['t_period']),restarts=list(np.cumsum(config['t_period'])[:-1]),last_epoch =self.epoch,ratio=config['restart_ratio'])
        elif config['one_cycle']:
            print('Using OneCycleLR_Restart Scheduler!')
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=config['train_steps'], epochs=config['nepoch'])

        else:
            self.scheduler = None
        self.criterion = criterion
        self.config = config
        
        self.mode = 'train'
        self.loss_list = config['loss_list']
        self.time_weight = None



    def cal_loss(self, pred_data, gt_data, type='recon',device='cuda',epoch=-1,weight=None):
        if type.find('recon') >-1: 
            pred_data = pred_data.reshape(gt_data.shape)

            if weight is None:
                loss = self.criterion['recon'](pred_data,gt_data)
            else:
                loss = self.criterion['recon'](pred_data,gt_data) * weight

                
            return loss



    def step(self, data,epoch=-1):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img, loss_dict
        '''
        gt_data = data['gt_img'].clone()
        output_list = self.forward_data(data)
        recon_img = output_list['recon_img']

        if self.config['method'] == 'ours_warp':
            output_list['middle_recon_gt'] = gt_data.clone()
        # cal_loss
        loss = torch.tensor(0.).to(self.config['device'])
        loss_dict = {}
        for key in self.loss_list:
            if key == 'recon':
                if not self.time_weight is None:
                    time_weight = self.time_weight.clone().reshape(1,-1).repeat(gt_data.shape[0],1).reshape(gt_data.shape[0],self.config['fut_len'],1,1,1)
                else:
                    time_weight = None
                loss_dict[key] = self.cal_loss(recon_img, gt_data.clone(), key,loss.device,weight=time_weight)
            
        for key in loss_dict.keys():
            loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()
        loss_dict['total'] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if 'pred_sim_matrix' in output_list:
            for i in range(len(output_list['pred_sim_matrix'])):
                output_list['pred_sim_matrix'][i] = output_list['pred_sim_matrix'][i].cpu().detach().numpy()
                output_list['gt_sim_matrix'][i] = output_list['vis_sim_matrix'][i].cpu().detach().numpy()
                output_list['vis_sim_matrix'][i] = None
        output_list['recon_img'] = output_list['recon_img'].cpu().detach()

        return output_list, loss_dict

    def val(self, data,epoch):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img, loss
        '''
        gt_data = data['gt_img'].clone()
        output_list = self.forward_data(data,update_sim_matrix=True)

        recon_img = output_list['recon_img']

        gt_data = gt_data.reshape(recon_img.shape)
        if self.config['method'] == 'ours_warp':
            output_list['middle_recon_gt'] = gt_data.clone()

        # cal_lossmid
        loss = torch.tensor(0.).to(self.config['device'])
        loss_dict = {}
        for key in self.loss_list:
            if key == 'recon':
                loss_dict[key] = self.cal_loss(recon_img, gt_data.clone(), key,loss.device)
            
        for key in loss_dict.keys():
            loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()
        loss_dict['total'] = loss.item()
        if 'pred_sim_matrix' in output_list:
            for i in range(len(output_list['pred_sim_matrix'])):
                    output_list['pred_sim_matrix'][i] = output_list['pred_sim_matrix'][i].cpu().detach().numpy().copy()
                    output_list['gt_sim_matrix'][i] = output_list['vis_sim_matrix'][i].cpu().detach().numpy().copy()
                    output_list['vis_sim_matrix'][i] = None
        output_list['recon_img'] = output_list['recon_img']

        return output_list, loss_dict
    
    def test(self, data,epoch):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img, loss
        '''
        gt_data = data['gt_img'].clone()
        data['input_img'] = data['input_img'][:,:self.config['prev_len']] # inference
        output_list = self.forward_data(data,update_sim_matrix=True,inference=True)
        
        output_list['recon_img'] = output_list['recon_img'].cpu().detach()

        return output_list

    def forward_data(self,data,update_sim_matrix=True,inference=False):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img_stack
        '''

        img_stack = data['input_img'] # N, H, W, C

        if self.config['long_term']:
            output_list = self.model.module.long_term_forward(img_stack)
        else:
            
            output_list = self.model(img_stack,inference=inference)

        return output_list




