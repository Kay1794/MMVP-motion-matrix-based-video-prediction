import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import math
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import cv2
import lpips
import random

lpips = lpips.LPIPS(net='alex')
metrics_to_save = []


def build_similarity_matrix(emb_feats,thre=-1,sigmoid=False,k=-1,cut_off=False):
    '''

    :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)
    :return: similarity matrix (N, T-1, h*w, h*w) current frame --> next frame
    '''
    B,T,c,h,w = emb_feats.shape
    emb_feats = emb_feats.permute(0,1,3,4,2) #  (B,T,h,w,c)
    normalize_feats = emb_feats / (torch.norm(emb_feats,dim=-1,keepdim=True)+1e-6) #  (B,T,h,w,c)
    prev_frame = normalize_feats[:,:T-1].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
    next_frame = normalize_feats[:,1:].reshape(-1,h*w,c,) # (B*(T-1),h*w,c)
    similarity_matrix = torch.bmm(prev_frame,next_frame.permute(0,2,1)).reshape(B,T-1,h*w,h*w) # (N*(T-1)*h*w)
    
    if cut_off:
        similarity_matrix = cut_off_process(similarity_matrix,thre,sigmoid,k)

    return similarity_matrix

def sim_matrix_postprocess(similar_matrix,config=None):
    B,T,hw1,hw2 = similar_matrix.shape

    similar_matrix = similar_matrix.reshape(similar_matrix.shape[0],similar_matrix.shape[1],-1)
    similar_matrix = F.softmax(similar_matrix,dim=-1)


    return similar_matrix.reshape(B,T,hw1,hw2)

def sim_matrix_interpolate(in_matrix,ori_hw,target_hw):

    ori_h,ori_w = ori_hw[0],ori_hw[1]
    target_h,target_w = target_hw[0],target_hw[1]
    B,T,hw,hw = in_matrix.shape
    ori_matrix = in_matrix.clone().reshape(B,T,ori_h,ori_w,ori_h,ori_w)
    ori_matrix_half = F.interpolate(ori_matrix.reshape(-1,ori_h,ori_w).unsqueeze(1),(target_h,target_w),mode='bilinear').squeeze(1) # (BThw,target_h,target_w)
    new_matrix = F.interpolate(ori_matrix_half.reshape(B,T,ori_h,ori_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(-1,ori_h,ori_w).unsqueeze(1),(target_h,target_w),mode='bicubic').squeeze(1) #(BT*targethw,target_h,target_w)
    new_matrix = new_matrix.reshape(B,T,target_h,target_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(B,T,target_h*target_w,target_h*target_w)

    return new_matrix

def cut_off_process(similarity_matrix,thre,sigmoid=False,k=-1):

    B = similarity_matrix.shape[0]
    T_prime = similarity_matrix.shape[1]
    hw = similarity_matrix.shape[2]
    new_similarity_matrix = similarity_matrix.clone()
    #mask all diagonal to zeros
    '''
    diagonal_mask = torch.zeros_like(new_similarity_matrix[0,0]).to(similarity_matrix.device).bool() #(h*w,h*w)
    diagonal_mask.fill_diagonal_(True)
    diagonal_mask = diagonal_mask.reshape(1,1,hw,hw).repeat(B,T_prime,1,1)
    new_similarity_matrix[diagonal_mask] = 0.
    '''
    if sigmoid:
        new_similarity_matrix[new_similarity_matrix<0] = 0.
        new_similarity_matrix = F.sigmoid(new_similarity_matrix)
        #similarity_matrix = F.sigmoid((similarity_matrix+1)/2.)
    elif k > -1: # select top k
        new_similarity_matrix[new_similarity_matrix<0.] = 0.
        select_num = int(new_similarity_matrix.shape[-1] * k)
        top_k,_ = torch.topk(new_similarity_matrix,select_num,dim=-1)
        thre_value = top_k[:,:,:,-1:]
        new_similarity_matrix[new_similarity_matrix<thre_value] = 0.
    else:
        new_similarity_matrix[new_similarity_matrix<thre] = 0.

    return new_similarity_matrix



def cum_multiply(value_seq,cum_softmax = False,reverse=True):
    '''

    :param value_seq: (B,S,***), B - batch num; S- sequence len
    :return: output value_seq: (B,S,***)
    '''
    #print(value_seq.shape)
    if not reverse: # reverse means last element is the one multiplied most times,i.e. the reference is the last element:
        value_seq = torch.flip(value_seq,dims=[1])
    B,T,hw,hw = value_seq.shape
    new_output = value_seq.clone()
    for i in range(value_seq.shape[1]-2,-1,-1):
        cur_sim = new_output[:, i].reshape(B,hw,hw).clone()
        next_sim = new_output[:,i+1].reshape(B,hw,hw).clone()
        new_output[:,i] = torch.bmm(cur_sim,next_sim).reshape(B,hw,hw)
    
    if not reverse:
        new_output = torch.flip(new_output,dims=[1])
    if cum_softmax:
        new_output = sim_matrix_postprocess(new_output)
    return new_output

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f',is_val=False):
        self.name = name
        self.fmt = fmt
        self.is_val = is_val
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        if self.is_val:
            fmtstr = 'val {name} {avg' + self.fmt + '}'

        return fmtstr.format(**self.__dict__)


def metric_print(metrics,epoch_num,step_num,t,last_iter=False):
    '''

    :param metrics: metric list
    :param epoch_num: epoch number
    :param step_num: step number
    :param t: time duration
    :return: string
    '''
    if last_iter:
        base = 'Epoch {} \t'.format(epoch_num)
    else:
        base = 'Epoch {} iter {}\t'.format(epoch_num, step_num)
    for key in metrics.keys():
        base = base + str(metrics[key]) + '\t'
    final = base + 'takes {} s'.format(t)
    return final


def update_metrics(metrics,loss_dict):
    for key in loss_dict.keys():
        metrics[key].update(loss_dict[key])

    return metrics


def img_valid(img):
    img = img + 0.5
    img = img
    img[img < 0] = 0.
    img[img > 1.] = 1.

    return img

def img_clamp(img):

    img[img < 0] = 0.
    img[img > 255] = 255
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.astype(np.uint8)

    return img

def torch_img_clamp_normalize(img):

    img[img < 0] = 0.
    img[img > 255] = 255
    img /= 255.

    return img

def MAE(true,pred):
    return (np.abs(pred-true)).sum()

def MSE(true,pred):
    return ((pred-true)**2).sum()



def visualization_check(save_path,epoch,image_list,valid=False,is_train=False):
    plt.clf()
    if is_train:
        save_file = save_path + '/visual_check_train_' + str(epoch) + '.png'
    else:
        save_file = save_path + '/visual_check_' + str(epoch) + '.png'
    sample_num = len(image_list)
    h1_image = []   #gt
    h2_image = [] #recon
    
    for i in range(sample_num):
        gt,recon = image_list[i]
        if valid:
            gt = img_clamp(gt)
            recon = img_clamp(recon)
        else:
            gt = img_valid(gt)
            recon = img_valid(recon)
        h1_image.append(gt)
        h2_image.append(recon)
        

    h1_image = np.hstack(h1_image)
    h2_image = np.hstack(h2_image)
    whole_image = np.vstack([h1_image,h2_image])
    plt.imshow(whole_image,interpolation="nearest")
    plt.savefig(save_file, dpi=400, bbox_inches ="tight", pad_inches = 1)

def visualization_check_video(save_path,epoch,image_list,valid=False,is_train=False,matrix=False,config=None,long_term=False):
    plt.clf()
    if is_train:
        save_file = save_path + '/visual_check_train_'
    else:
        save_file = save_path + '/visual_check_'

    if matrix:
        save_file = save_file + 'matrix_'
    save_file = save_file + str(epoch) + '.png'
    sample_num = len(image_list)
    vis_image = []
    dtf_image = []
    diff_prev_image = [] # diff_prev
    diff_gt_image = [] # diff_gt
    for i in range(sample_num):
        gt_seq,recon_seq = image_list[i]
        
        if matrix:
            h= config['mat_size'][0][0]
            w = config['mat_size'][0][1]
            gt_seq = gt_seq.reshape(h,w,h,w)
            recon = recon_seq.reshape(h,w,h,w)
            select_index = [[h//4,w//4],[h//4,w*3//4],[h//2,w//2],[h*3//4,w//4],[h*3//4,w*3//4]]
           # select_index = [[5,5],[5,25],[15,15],[25,5],[25,25]]
            vis_image_row = []
            #dtf_image_row = []
            for index in select_index:
                vis_image_row.append(np.hstack([np.log(gt_seq[index[0],index[1]]),np.log(recon[index[0],index[1]]),np.log(gt_seq[index[0],index[1]]-recon[index[0],index[1]])]))
            vis_image_row = np.hstack(vis_image_row)
            #dtf_image_row = np.hstack(dtf_image_row)
            vis_image.append(vis_image_row)
            #dtf_image.append(dtf_image_row)


        else:
            
           
            if gt_seq.shape[0] > 20 or long_term:

                gt = np.hstack(gt_seq)
                recon = np.hstack(recon_seq)
                if valid:
                    gt = img_clamp(gt)
                    recon = img_clamp(recon)
                else:
                    gt = img_valid(gt)
                    recon = img_valid(recon)
                recon_range = np.zeros_like(gt)
                recon_range[:,-recon.shape[1]:,:] = recon
                recon = recon_range
                vis_image.append(np.vstack([gt,recon]))
                
            else:
                gt = np.hstack(gt_seq[-2:])
            
                recon = np.hstack(recon_seq)
                if valid:
                    gt = img_clamp(gt)
                    recon = img_clamp(recon)
                else:
                    gt = img_valid(gt)
                    recon = img_valid(recon)

            
                overlay = recon.copy()
                gt_frame = img_clamp(gt_seq[-1]) if valid else img_valid(gt_seq[-1])
                last_frame = img_clamp(gt_seq[-2]) if valid else img_valid(gt_seq[-2])
                
                overlay_last = last_frame*.5 + overlay *.5
                overlay_gt = gt_frame*.5 + overlay *.5
                real_overlay = gt_frame *.5 + last_frame *.5 
                if valid:
                    overlay_last = overlay_last.astype(np.uint8)
                    overlay_gt = overlay_gt.astype(np.uint8)
                    real_overlay = real_overlay.astype(np.uint8)
                vis_image.append(np.hstack([gt,recon,overlay_last,overlay_gt,real_overlay]))
        #exit()
    
    whole_image = np.vstack(vis_image)
    if matrix:
        plt.imshow(whole_image,interpolation="nearest",cmap='hot')
    else:
        if whole_image.shape[-1] == 1:
            plt.imshow(whole_image,interpolation="nearest",cmap='gray')
        else:
            plt.imshow(whole_image,interpolation="nearest")
    plt.axis('off')
    plt.savefig(save_file, dpi=400, bbox_inches ="tight", pad_inches = 0)


def visualization_check_video_testmode(save_path,image_list,valid=False,config=None,iter_id=0):
    plt.clf()
    check_folder(save_path)

    save_file = save_path + '/visual_iter_'

    save_file = save_file + str(iter_id) + '.png'
    prev_seq,gt_seq,recon_seq= image_list
    sample_num = len(prev_seq)
    vis_image = []
    for i in range(sample_num):
        
        last_frame = img_clamp(prev_seq[i][-1]) if valid else img_valid(prev_seq[i][-1])
        gt_frame = img_clamp(gt_seq[i][0]) if valid else img_valid(gt_seq[i][0])
        recon_frame = img_clamp(recon_seq[i][0]) if valid else img_valid(recon_seq[i][0])
        cur_prev = img_clamp(np.hstack(prev_seq[i])) if valid else img_valid(np.hstack(prev_seq[i]))
        cur_gt = img_clamp(np.hstack(gt_seq[i])) if valid else img_valid(np.hstack(gt_seq[i]))
        cur_recon = img_clamp(np.hstack(recon_seq[i])) if valid else img_valid(np.hstack(recon_seq[i]))
        
        
        overlay_last = last_frame*.5 + recon_frame *.5
        overlay_gt = gt_frame*.5 + recon_frame *.5
        real_overlay = gt_frame *.5 + last_frame *.5 
        if valid:
            overlay_last = overlay_last.astype(np.uint8)
            overlay_gt = overlay_gt.astype(np.uint8)
            real_overlay = real_overlay.astype(np.uint8)
        vline = np.zeros((cur_prev.shape[0],20,3)).astype(np.uint8)
        vis_image.append(np.hstack([cur_prev,vline,cur_gt,cur_recon,vline,overlay_last,overlay_gt,real_overlay]))
        #exit()

    whole_image = np.vstack(vis_image)
    if whole_image.shape[-1] == 1 :
        plt.imshow(whole_image,interpolation="nearest",cmap='gray')
    else:
        plt.imshow(whole_image,interpolation="nearest")
    plt.axis('off')
    plt.savefig(save_file, dpi=400, bbox_inches ="tight", pad_inches = 0)


def image_evaluation(image_list,gt_image_list,eval_metrics,valid=False):

    
    size = image_list.shape
    if len(size) > 4:
        image_list = image_list.reshape(size[0]*size[1],size[2],size[3],size[4])
    size = gt_image_list.shape
    if len(size) > 4:
        gt_image_list = gt_image_list.reshape(size[0]*size[1],size[2],size[3],size[4])
    for i in range(image_list.shape[0]):
        if valid:
            image = img_clamp(image_list[i]) /255.
            gt_image = img_clamp(gt_image_list[i]) / 255.
        else:
            image = img_valid(image_list[i])
            gt_image = img_valid(gt_image_list[i])

        for key in eval_metrics:
            
            if key == 'psnr':
                #metrics_to_save.append(psnr(gt_image.copy(),image.copy()))
                eval_metrics[key].update(psnr(gt_image.copy(),image.copy()))
            elif key == 'ssim':
                eval_metrics[key].update(ssim(gt_image.copy(),image.copy(),multichannel=True))
            elif key == 'mae':
                eval_metrics[key].update(MAE(gt_image.copy(),image.copy()))
            elif key == 'mse':
                eval_metrics[key].update(MSE(gt_image.copy(),image.copy()))
            elif key == 'lpips':
                eval_metrics[key].update(lpips(torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0),torch.from_numpy(gt_image.copy()).permute(2,0,1).unsqueeze(0)).item())

    return eval_metrics

class VGG_feature(nn.Module):
    def __init__(self,device):
        super(VGG_feature, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)

        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, input):
        '''
        input: B,T,c,H,W
        output: B,T,c,H//8,W//8
        '''
        B,T,C,H,W = input.shape
        input = input.reshape(-1,C,H,W)
        with torch.no_grad():
            vgg_output = self.vgg16_conv_4_3(input.clone())
        X,c,h,w = vgg_output.shape
        vgg_output = vgg_output.reshape(B,T,c,h,w)

        return vgg_output


class VGG_loss(nn.Module):
    def __init__(self,device,reduction='mean'):
        super(VGG_loss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        self.reduction = reduction
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt,norm=False):
        
        if len(output.shape) > 4:
            output = output.reshape(-1,output.shape[-3],output.shape[-2],output.shape[-1])
        if len(gt.shape) > 4:
            gt = gt.reshape(-1,output.shape[-3],output.shape[-2],output.shape[-1])
        if output.shape[1] != 3:

            output = output.permute(0,3,1,2)
            gt = gt.permute(0,3,1,2)
        if not norm:
            output = torch_img_clamp_normalize(output)
            gt = torch_img_clamp_normalize(gt)
        else:
            gt += 0.5
            output += 0.5


        vgg_output = self.vgg16_conv_4_3(output.clone())
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())
        
        if self.reduction == 'sum':
            loss =torch.sum(F.mse_loss(vgg_output, vgg_gt,reduction='sum')) / (output.shape[0])
        else:
            loss =F.mse_loss(vgg_output, vgg_gt,reduction='mean')


        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-6,reduction='sum'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y,weight=None,reduction='sum'):
        diff = (x - y)
        if self.reduction == 'sum':
            if weight is None:
                loss = torch.sum(torch.sqrt(diff * diff + self.eps))
            else:
                loss = torch.sum(torch.sqrt(diff * diff + self.eps)*weight)
            
            loss /= x.shape[0] #batch mean
        elif self.reduction == 'mean':
            if weight is None:
                loss = torch.mean(torch.sqrt(diff * diff + self.eps)) 
            else:
                loss = torch.mean(torch.sqrt(diff * diff + self.eps)*weight)

        
        return loss

class JSDLoss(nn.Module):
    """JSD Loss"""

    def __init__(self, weight=1.):
        super(JSDLoss, self).__init__()
        self.weight = weight

    def forward(self, feat_1,feat_2):
        c = feat_1.shape[1]
        BT = feat_1.shape[0]
        feat_1 = feat_1.permute(0,2,3,1).reshape(-1,c)
        feat_2 = feat_2.permute(0,2,3,1).reshape(-1,c)
        feat_1 = F.softmax(feat_1,dim=-1)
        feat_2 = F.softmax(feat_2,dim=-1)
        p_mixture = torch.clamp((feat_1 + feat_2) / 2., 1e-7, 1).log()
        loss = self.weight * (F.kl_div(p_mixture, feat_1, reduction='batchmean') +
        F.kl_div(p_mixture, feat_2, reduction='batchmean')) / 2. * BT

        #print(loss)
                
        return loss


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=1e-6, last_epoch=-1,ratio=0.5):
        self.T_period = list(T_period)
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = list(restarts)
        self.restart_weights = [ratio ** (i+1) for i in range(len(restarts))]
        self.last_restart = 0
        print('restart ratio: ',ratio,' T_period: ',T_period,' minimum lr: ',eta_min)
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[list(self.restarts).index(self.last_epoch) + 1]
            weight = self.restart_weights[list(self.restarts).index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]