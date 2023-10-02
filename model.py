import torch
from torchvision.models.resnet import *
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from layers import *
import math
import time
from utils import *
import h5py

#from simvp_model import OriInception

#hf = h5py.File('./matrix_data.h5', 'w')

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bilinear = config['bilinear']
        self.prev_len = config['prev_len']
        self.fut_len = config['fut_len']
        self.available_pred_len = 1 if config['long_term'] else config['fut_len'] # currently only support pred 1 future frame at a time

        self.n_channel = config['n_channel']
        if config['shuffle_setting']:
            self.unshuffle = nn.PixelUnshuffle(2)
            self.shuffle = nn.PixelShuffle(2)

        self.encoder = RRDBEncoder(config)
        self.decoder = RRDBDecoder(config)

        # Similarity projection
        self.filter_block_proj = []

        factor = 2

        translate_factor = config['translate_factor']
        

        if config['filter_block']:    
            
            #highres feat
            high_scale = len(self.config['downsample_scale'])-1
            feat_len = self.config['base_channel']*(2**(high_scale))
            self.filter_block_proj.append(nn.Sequential(nn.Conv2d(feat_len,self.config['base_channel'] // translate_factor ,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] // translate_factor),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel'] // translate_factor,self.config['base_channel'] // translate_factor ,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] // translate_factor),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel'] // translate_factor,self.config['base_channel'] // translate_factor ,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] // translate_factor),
            nn.LeakyReLU()))

            #lowres feat
            low_scale = high_scale + 1
            feat_len = self.config['base_channel']* (2**(low_scale))
            self.filter_block_proj.append(nn.Sequential(nn.Conv2d(feat_len,self.config['base_channel'] * 2 // translate_factor,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] * 2 // translate_factor),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel'] * 2 // translate_factor,self.config['base_channel'] * 2 // translate_factor,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] * 2 // translate_factor),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel'] * 2 // translate_factor,self.config['base_channel'] * 2 // translate_factor,kernel_size=3,padding=1), 
            nn.BatchNorm2d(self.config['base_channel'] * 2 // translate_factor),
            nn.LeakyReLU()))

        self.filter_block_proj = nn.ModuleList(self.filter_block_proj)


        #Motion Predictor

        if config['filter_block']:
            self.scale_fuser_1 = Up(config,self.config['base_channel'] * 2 // translate_factor, self.config['base_channel'] // translate_factor , self.bilinear, scale=2)
            self.scale_fuser_2 = nn.Sequential(nn.Conv2d(self.config['base_channel'] // translate_factor,self.config['base_channel'] // translate_factor,kernel_size=3,padding=1),
            nn.BatchNorm2d(self.config['base_channel']// translate_factor),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel']// translate_factor,self.config['base_channel']// translate_factor ,
            kernel_size=3,padding=1),
            nn.BatchNorm2d(self.config['base_channel']// translate_factor),
            nn.LeakyReLU())  if self.config['motion_use_bn']  else\
            nn.Sequential(nn.Conv2d(self.config['base_channel']// translate_factor,self.config['base_channel']// translate_factor,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.config['base_channel']// translate_factor,self.config['base_channel']// translate_factor ,
            kernel_size=3,padding=1),
            nn.LeakyReLU())
        else:
            self.scale_fuser_1 = Up(config,self.config['base_channel'] * 4 , self.config['base_channel'] * 2, self.bilinear, scale=2)
            self.scale_fuser_2 = nn.Sequential()
        
        

        self.predictor = PredictModel(config=self.config,hidden_len=self.config['pred_base_channel'],mx_h=config['mat_size'][0][0],mx_w=config['mat_size'][0][1])
            

        
        self.feat_shuffle = []
        self.feat_unshuffle = []
        self.feat_scale_list = []

        for i in range(len(config['downsample_scale'])-1):
            feat_shuffle_scale = 1
            for s in range(len(config['downsample_scale'])-2,i-1,-1):
                feat_shuffle_scale *= config['downsample_scale'][s]
            self.feat_scale_list.append(feat_shuffle_scale)
            self.feat_shuffle.append(nn.PixelShuffle(feat_shuffle_scale))
            self.feat_unshuffle.append(nn.PixelUnshuffle(feat_shuffle_scale))
        self.feat_shuffle = nn.ModuleList(self.feat_shuffle)
        self.feat_unshuffle = nn.ModuleList(self.feat_unshuffle)

        #-----------------------------------------------------------------------------------------------------------------------------------#
        if self.config['res_cat_img']:

            feat_factor = 2**((len(self.config['downsample_scale'])-1))
            base_channel = self.config['base_channel']

            res_shuffle_scale = 1
            for s in range(len(config['downsample_scale'])-1):
                res_shuffle_scale *= config['downsample_scale'][s]

            self.res_shuffle = nn.PixelShuffle(res_shuffle_scale // self.config['res_img_scale'])
            self.res_unshuffle = nn.PixelUnshuffle(res_shuffle_scale // self.config['res_img_scale'])
            self.res_shuffle_scale = res_shuffle_scale


        self.enhancer = ImageEnhancer(config=self.config,n_channels=self.config['n_channel'] , base_channel=self.config['base_channel'] //2 )
        
    def feat_generator_window(self,feats, sim_matrix,kernel):
        '''

        :param feats: [B,T,c,h,w]
        :param sim_matrix: [B,T,h*w,ws*ws]
        '''
        B, T, c, h, w = feats.shape
        window_size = kernel[0] * kernel[1]
        h_w,w_w = kernel
        feats = feats.permute(0,3,4, 1, 2)  # (B,h,w,c,T)
        feats = feats.reshape(B*h*w, T, c)  # (B*h*w,Prev T,c)

        _,_,hw,ws2 = sim_matrix.shape
        sim_matrix = sim_matrix.permute(0,2,3,1).reshape(B*hw, window_size, T) # Batch*H*W, ws**2,Prev T

        weight = torch.sum(sim_matrix, dim=-1).reshape(-1, window_size,1) + 1e-6
        new_feats = torch.bmm(sim_matrix, feats)/ weight # Batch*H*W,ws**2,c
        
        new_feats = new_feats.reshape(B,h,w,h_w,w_w,c)
        new_feats = fold_feature_patch(new_feats,kernel)


        return new_feats

    def feat_generator(self, feats, sim_matrix,feat_idx,img_compose=False,scale=1):
        '''

        :param feats: [B,T,c,h,w]
        :param sim_matrix: [B,T,h*w,h*w]
        :return: new_feats: [B,c,h,w]
        '''
        B, T, c, h, w = feats.shape
        # only test single motion
        if scale > 1: # if hw_cur != hw_target, only use the last sim matrix
            feats = feats[:,-1:,]
            sim_matrix = sim_matrix[:,-1:]
            T = 1
        feats = feats.permute(0, 2, 1, 3, 4)  # (B,c,T,h,w)
        feats = feats.reshape(B, c, T * h * w).permute(0, 2, 1)  # (B,Prev T*h*w,c)
        B,T,hw_cur,hw_target = sim_matrix.shape
        sim_matrix = sim_matrix.reshape(B, T * hw_cur, hw_target).permute(0, 2, 1) # Batch, fut H*W, Prev T*HW
        weight = torch.sum(sim_matrix, dim=-1).reshape(-1, 1, hw_target) + 1e-6
        new_feats = torch.bmm(sim_matrix, feats).permute(0, 2, 1) / weight
        new_feats = new_feats.reshape(B, c, h*scale, w*scale)


        return new_feats

    def feat_compose(self, emb_feat_list, sim_matrix,img_compose=False,scale=1,use_gt=False):
        '''

        :param emb_feat_list: (scale_num, (B,T,c,h,w))
        :param sim_matrix:  (B,T-1,h,w,h,w)
        :param use_gt_sim_matrix: bool
        :return: fut_emb_feat_list (scale_num, (B,t,c,h,w))
        '''
        fut_emb_feat_list = []
        ori_emb_feat_list = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                fut_emb_feat_list.append(None)
                ori_emb_feat_list.append(None)
                continue

            fut_emb_feat_list.append([])
            cur_emb_feat = emb_feat_list[i].clone()
            ori_emb_feat_list.append(torch.mean(emb_feat_list[i].clone(),dim=1))
            
            sim_matrix_seq = sim_matrix[i].clone()
            B = sim_matrix_seq.shape[0]
            N, c, h, w = cur_emb_feat.shape
            cur_emb_feat = cur_emb_feat.reshape(B,-1,c,h,w)
            cur_emb_feat = cur_emb_feat[:,:self.prev_len] if (not use_gt) else cur_emb_feat.clone()

            for t in range(self.available_pred_len):
                active_matrix_seq = sim_matrix_seq[:,:(self.prev_len-1)]
                if t > 0:
                    fut_t_matrix =sim_matrix_seq[:,(self.prev_len+t-1):(self.prev_len+t)].clone()
                else:
                    fut_t_matrix = sim_matrix_seq[:,(self.prev_len-1):(self.prev_len+t)].clone()
                active_matrix_seq = torch.cat([active_matrix_seq,fut_t_matrix],dim=1)
                


                cur_sim_matrix = cum_multiply(active_matrix_seq.clone())  # B, T+1, h,w,h,w
                composed_t_feats = self.feat_generator(cur_emb_feat[:, :(self.prev_len)].clone(),
                                                        cur_sim_matrix,feat_idx=i,img_compose=img_compose,scale=scale)
                                                    
                fut_emb_feat_list[i].append(composed_t_feats.clone())
                # update future frame features in the emb_feat_list
                if (not use_gt):
                    if scale == 1:
                        if  cur_emb_feat.shape[1] > (self.prev_len+t):
                            cur_emb_feat[:,t+self.prev_len] = composed_t_feats.clone()
                        else:
                            cur_emb_feat = torch.cat([cur_emb_feat,composed_t_feats.clone().unsqueeze(1)],dim=1) #cat compose features for next frame prediction


            temp = torch.stack(fut_emb_feat_list[i], dim=1)
            
            fut_emb_feat_list[i] = temp.reshape(-1, c, h*scale, w*scale) # B*T,c,h,w


        return fut_emb_feat_list,ori_emb_feat_list



    def long_term_forward(self, input_image):
        output_list = {}
        pred_img_list = []
        B, T, H, W, C = input_image.shape
        cur_input_seq = input_image.clone()[:,:self.prev_len]
        
        for i in range(self.fut_len):

            cur_output = self.forward(cur_input_seq,inference=True)
            pred_img = cur_output['recon_img'].reshape(B,-1,H,W,C)
            pred_img_list.append(pred_img.clone())
            cur_input_seq = torch.cat((cur_input_seq[:,1:],pred_img),dim=1)

        '''
        prepare for output
        '''
        output_list['recon_img'] = torch.cat(pred_img_list,dim=1)

        return output_list


    def forward(self, input_image,update_sim_matrix=True,inference=False):

        ori_input = input_image.clone()
        output_list = {}

        B, T, H, W, C = input_image.shape
        
        ori_input_image = input_image.clone()
        input_image = input_image.reshape(-1, H, W, C)  # B*T,H,W,C
        input_image = input_image.permute(0, 3, 1, 2)
        if self.config['shuffle_setting']:
            input_image = self.unshuffle(input_image)
        input_image_raw = input_image.clone()
        raw_img_wh = input_image_raw.shape[-2:]

        emb_feat_list = self.encoder(input_image) # N, C, H, W

        prev_feat_list = []
        for s in [-2,-1]:
            n,c,h,w = emb_feat_list[s].shape
            prev_feat_list.append(emb_feat_list[s].reshape(B,T,c,h,w)[:,:self.prev_len].reshape(-1,c,h,w))
        if 'prev_recon' in self.config['loss_list']:
            prev_recon_img = self.decoder(prev_feat_list)
            if self.config['shuffle_setting']:
                prev_recon_img = self.shuffle(prev_recon_img)
            output_list['prev_recon'] = prev_recon_img.clone().permute(0, 2, 3,1)

    

        '''
        Filter Block
        '''
        self.feat_res = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                self.feat_res.append(None)
            else:
                self.feat_res.append(emb_feat_list[i].shape[2:])

        sim_feat_list = []
        
        for s in [-2,-1]:
            feat_hw = emb_feat_list[s].shape[-1] * emb_feat_list[s].shape[-2]
            mat_hw = self.config['mat_size'][s][-1] * self.config['mat_size'][s][-2]
            if mat_hw != feat_hw:
                    sim_feat = F.interpolate(emb_feat_list[s].clone(),size=tuple(self.config['mat_size'][s]),mode='bilinear')
            else:
                sim_feat = emb_feat_list[s].clone()
            if self.config['filter_block']:
                sim_feat = self.filter_block_proj[s](sim_feat)
                
                sim_feat_list.append(sim_feat.clone())
            else:
                sim_feat_list.append(emb_feat_list[s].clone())
    

        '''
        construct similarity matrix
        '''

        similar_matrix = []
        teacher_feat_list = []
        student_feat_list = []
        gt_sim_matrix = []
        vis_sim_matrix = []
        prev_sim_matrix = []
        
        
        for i in [-2,-1]:
            high_res = None
            N = sim_feat_list[i].shape[0]   
            h = sim_feat_list[i].shape[2]
            w = sim_feat_list[i].shape[3]

            cur_sim_matrix = build_similarity_matrix(sim_feat_list[i].clone().reshape(B, T, -1, h, w))
            prev_sim_matrix.append(cur_sim_matrix[:,:(self.prev_len-1)].clone())     

            if not inference:
                gt_sim_matrix.append(sim_matrix_postprocess(cur_sim_matrix[:,(self.prev_len-1):].clone(),config=self.config))
                vis_sim_matrix.append(sim_matrix_postprocess(cur_sim_matrix.clone(),config=self.config)) #complete sim mat list 

        
        output_list['gt_sim_matrix'] = gt_sim_matrix
        output_list['vis_sim_matrix'] = vis_sim_matrix

        '''
        Predict next similarity matrix
        '''

        
        res_feat = None
        pred_sim_matrix = [ None,None]
        if self.config['pred_type'] == 'highonly':
            
            pred_fut_matrix,_ = self.predictor(prev_sim_matrix[0],softmax=False,res=None)
            pred_sim_matrix[0] = pred_fut_matrix.clone()

            pred_sim_matrix[1] = sim_matrix_interpolate(pred_fut_matrix.clone(),self.config['mat_size'][0],self.config['mat_size'][1]).clone()

            
        
        # post process the matrix
        pred_sim_matrix[0] = sim_matrix_postprocess(pred_sim_matrix[0])
        pred_sim_matrix[1] = sim_matrix_postprocess(pred_sim_matrix[1])
        output_list['pred_sim_matrix'] = pred_sim_matrix
        
            
        # update similarity matrix list
        for i in range(len(prev_sim_matrix)):


            
            new_cur_sim_matrix = torch.cat([sim_matrix_postprocess(prev_sim_matrix[i].clone(),config=self.config),pred_sim_matrix[i].clone()],dim=1)
            similar_matrix.append(new_cur_sim_matrix.clone())



        '''
        compose feature
        '''
        compose_feat_list = []
        similar_matrix_for_compose = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                compose_feat_list.append(None)
                similar_matrix_for_compose.append(None)
                continue
            if i <  (len(emb_feat_list)-2):
                h,w = emb_feat_list[i].shape[-2:]
                target_size = (h//self.feat_scale_list[i] *  self.feat_scale_list[i],w//self.feat_scale_list[i] *  self.feat_scale_list[i])
                cur_feat = self.feat_unshuffle[i](F.interpolate(emb_feat_list[i].clone(),size=target_size,mode='bilinear'))

                if (cur_feat.shape[-2] != self.config['mat_size'][0][-2]) or (cur_feat.shape[-1] != self.config['mat_size'][0][-1]): 
                    compose_feat_list.append(F.interpolate(cur_feat,size=tuple(self.config['mat_size'][0]),mode='bilinear'))
                else:
                    compose_feat_list.append(cur_feat.clone())
                
                similar_matrix_for_compose.append(similar_matrix[0].clone())
                
            else:
                if (emb_feat_list[i].shape[-2] != self.config['mat_size'][i-len(emb_feat_list)+2][-2]) \
                    or (emb_feat_list[i].shape[-1] != self.config['mat_size'][i-len(emb_feat_list)+2][-1]): 
                    compose_feat_list.append(F.interpolate(emb_feat_list[i].clone(),size=tuple(self.config['mat_size'][i-len(emb_feat_list)+2]),mode='bilinear'))
                else:
                    compose_feat_list.append(emb_feat_list[i].clone())
        
        similar_matrix_for_compose.append(similar_matrix[0])
        similar_matrix_for_compose.append(similar_matrix[1])

    

        compose_fut_feat_list,avg_emb_feat_list = self.feat_compose(compose_feat_list, similar_matrix_for_compose)

        
        
        '''
        decode feature
        '''
        for i in range(len(compose_fut_feat_list)):
            if compose_fut_feat_list[i] is None:
                continue
            if i <  (len(emb_feat_list)-2):
                compose_fut_feat_list[i] = self.feat_shuffle[i](compose_fut_feat_list[i].clone())
            if (compose_fut_feat_list[i].shape[-2] != self.feat_res[i][-2]) or (compose_fut_feat_list[i].shape[-1] != self.feat_res[i][-1]):
                compose_fut_feat_list[i] = F.interpolate(compose_fut_feat_list[i].clone(),size=tuple(self.feat_res[i]),mode='bilinear')
            
        recon_img = self.decoder(compose_fut_feat_list)

        '''
        add residual info
        '''

        final_recon_img = recon_img.clone()
        
            
        if self.config['res_cat_img']:
            
            # build compose image

            if raw_img_wh != recon_img.shape[2:]:
                # input_image_raw = F.interpolate(input_image_raw,recon_img.shape[2:])
                #for davis
                std_w = int(self.config['mat_size'][0][0]  * self.res_shuffle_scale)
                std_h = int(self.config['mat_size'][0][1]  * self.res_shuffle_scale)

                input_image_raw = F.interpolate(input_image_raw,(std_w,std_h))
            

            image_list = [self.res_unshuffle(input_image_raw)]

            compose_image,avg_image = self.feat_compose(image_list, [similar_matrix[0]])
            compose_image = compose_image[0]

            compose_image = self.res_shuffle(compose_image) 
            if self.config['shuffle_setting']:
                fut_img_seq = self.shuffle(compose_image.clone())
            else:
                fut_img_seq = compose_image.clone()

            
            
            
        '''
        prepare for output
        '''

        if self.config['shuffle_setting']:
            recon_img = self.shuffle(recon_img)
            final_recon_img = self.shuffle(final_recon_img)
            if self.config['res_cat_img'] and (fut_img_seq.shape[2:] != final_recon_img.shape[2:]):
                fut_img_seq = F.interpolate(fut_img_seq,final_recon_img.shape[2:])

        final_recon_img = self.enhancer(torch.cat([final_recon_img.clone(),fut_img_seq.clone()],dim=1) if (self.config['res_cat_img']) else final_recon_img.clone())

        if recon_img.shape[-2] != H or recon_img.shape[-1] != W:
            recon_img = F.interpolate(recon_img,(H,W))
            final_recon_img = F.interpolate(final_recon_img,(H,W))

            
        recon_img = recon_img.permute(0, 2, 3, 1)
        final_recon_img = final_recon_img.permute(0,2,3,1)
        output_list['recon_img'] = final_recon_img
        
        output_list['middle_recon_img'] = recon_img.reshape(final_recon_img.shape).clone()
        output_list['teacher_feat'] = teacher_feat_list
        output_list['student_feat'] = student_feat_list

        return output_list
