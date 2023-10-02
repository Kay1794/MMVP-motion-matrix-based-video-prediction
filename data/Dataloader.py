import sys
import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import time
import cv2
from tqdm import tqdm
from matplotlib import pyplot
import warnings
from multiprocessing import Manager
from matplotlib import pyplot as plt
from data.data_utils import *
import configs
import math
from copy import deepcopy as cp
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import hickle as hkl
import shutil
import h5py
import gzip
import torchvision


class UCFPredDataset(Dataset):

    def __init__(self, config, split, is_train=False):
        self.config = config
        self.split = split
        self.is_train = is_train
        self.data_root = config['dataroot']+'/ucf_prediction_4to1_filter'

        if self.split == 'train':
            print('Loading train dataset')
            self.data_root = (self.data_root + '_train/').replace('filter','filter')
            self.data_source = [os.path.join(self.data_root, filename) for filename in os.listdir(self.data_root) if
                                filename.find('.npy') > -1]
            print('Loading train dataset finished, with size:', len(self.data_source))
        else:
            print('Loading test dataset')
            self.data_root = (self.data_root + '_test/')
            #.replace('filter','filter_dynamic')

            if config['mode'] in ['test','val'] and ( config['fut_len'] == 1):
              self.data_root = self.data_root.replace('filter','filter_'+config['val_subset'])

            self.data_source = sorted([os.path.join(self.data_root,filename) for filename in os.listdir(self.data_root) if
                                filename.find('.npy') > -1])
            print('Loading test dataset finished, with size:', len(self.data_source))
        print('Loading ', self.split, ' split: ', len(self.data_source), ' sequences')

    def __len__(self):
        return len(self.data_source)

    def preprocess_img(self, img_list):
        T,H,W,C = img_list.shape
        img_list = img_list.reshape(H*T,W,C)
        img_list = cv2.cvtColor(img_list,cv2.COLOR_BGR2RGB).reshape(T,H,W,C)
        if self.config['range'] == 1.:
            norm_img_list = img_list / 255.
            norm_img_list_tensor = norm_img_list- 0.5
        else:
            norm_img_list_tensor = img_list

        return norm_img_list_tensor

    def __getitem__(self, idx):
        #idx = 11
        cur_data = np.load(self.data_source[idx], allow_pickle=True).item()
        prev_frames_tensor = self.preprocess_img(cur_data['prev_frames'])
        fut_frames_tensor = self.preprocess_img(cur_data['fut_frames'])[:self.config['fut_len']]

        # print(fut_frames_tensor.shape)
        # exit()
        fut_frames_tensor = torch.from_numpy(fut_frames_tensor.copy()).float().to(self.config['device'])
        prev_frames_tensor = torch.from_numpy(prev_frames_tensor.copy()).float().to(self.config['device'])


        if self.split == 'train' and self.config['flip_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[1])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[1])
            flag = random.uniform(0,1)
            if flag < 0.5:
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])

        if self.split == 'train' and self.config['rot_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                k = random.randint(1, 3)
                prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(1,2),k=k)
                fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(1,2),k=k)

        prev_frames_tensor = torch.cat([prev_frames_tensor,fut_frames_tensor],dim=0)
        
        return prev_frames_tensor, fut_frames_tensor


class STRPM_UCFPredDataset(Dataset):



    def __init__(self, config, split, is_train=False):
        self.split = split
        self.config = config
        self.h5_path = config['dataroot']+'/ucf_prediction_official'
        if self.split == 'train':
            print('Loading train dataset')
            self.dataset = h5py.File(self.h5_path+'_'+split+'.h5', "r")
            print('Loading train dataset finished, with size:', len(self.dataset.keys()))
        else:
            print('Loading test dataset')
            self.dataset = h5py.File(self.h5_path + '_test.h5', "r")
            print('Loading test dataset finished, with size:', len(self.dataset.keys()))

    def preprocess_img(self, img_list):
        T,H,W,C = img_list.shape
        img_list = img_list.reshape(H*T,W,C)
        img_list = cv2.cvtColor(img_list,cv2.COLOR_BGR2RGB).reshape(T,H,W,C)
        if self.config['range'] == 1.:
            norm_img_list = img_list / 255.
            norm_img_list_tensor = norm_img_list- 0.5
        else:
            norm_img_list_tensor = img_list

        return norm_img_list_tensor

    def __len__(self):
        return len(self.dataset.keys())


    def __getitem__(self, idx):

        data_slice = np.asarray(self.dataset.get(str(idx)))

        sample = self.preprocess_img(data_slice)
        prev_frames_tensor = sample.copy()[:self.config['prev_len']]
        fut_frames_tensor = sample.copy()[self.config['prev_len']:(self.config['prev_len']+self.config['fut_len'])]

        fut_frames_tensor = torch.from_numpy(fut_frames_tensor.copy()).float().to(self.config['device'])
        prev_frames_tensor = torch.from_numpy(prev_frames_tensor.copy()).float().to(self.config['device'])


        if self.split == 'train' and self.config['flip_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[1])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[1])
            flag = random.uniform(0,1)
            if flag < 0.5:
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])

        if self.split == 'train' and self.config['rot_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                k = random.randint(1, 3)
                prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(1,2),k=k)
                fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(1,2),k=k)
        prev_frames_tensor = torch.cat([prev_frames_tensor,fut_frames_tensor],dim=0)

        return prev_frames_tensor,fut_frames_tensor


class KTHPredDataset(Dataset):
    def __init__(self, config, split,is_train=True):
        super(KTHPredDataset,self).__init__()
        self.config = config
        self.split = split
        if config['fut_len'] == 40:
            data = hkl.load(config['dataroot']+'/'+str(split)+'_data_gzip_t=40.hkl')
            indices = hkl.load(config['dataroot']+'/'+str(split)+'_indices_gzip_t=40.hkl')
        else:
            data = hkl.load(config['dataroot']+'/'+str(split)+'_data_gzip.hkl')
            indices = hkl.load(config['dataroot']+'/'+str(split)+'_indices_gzip.hkl')

        self.datas = data.swapaxes(2, 3).swapaxes(1,2)
        self.indices = indices
        self.pre_seq_length = config['prev_len']
        self.aft_seq_length = config['fut_len']
        self.transform = torchvision.transforms.Resize((64,64))
        
        

    
    def __len__(self):
        return len(self.indices)

    def preprocess_img(self, img_list):
        if self.config['range'] == 1.:
            norm_img_list = img_list / 255.
            norm_img_list_tensor = norm_img_list- 0.5
        else:
            norm_img_list_tensor = img_list
        

        return norm_img_list_tensor

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length

        prev_frames_tensor = torch.from_numpy(self.preprocess_img(self.datas[begin:end1,::].copy())).float().to(self.config['device']).permute(0,2,3,1)
        fut_frames_tensor = torch.from_numpy(self.preprocess_img(self.datas[end1:end2,::].copy())).float().to(self.config['device']).permute(0,2,3,1)
        
        if self.config['in_res'][0] == 64:
            prev_frames_tensor = self.transform(prev_frames_tensor.permute(0,3,1,2)).permute(0,2,3,1)
            fut_frames_tensor = self.transform(fut_frames_tensor.permute(0,3,1,2)).permute(0,2,3,1)


        if self.split == 'train' and self.config['flip_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[1])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[1])
            flag = random.uniform(0,1)
            if flag < 0.5:
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])

        if self.split == 'train' and self.config['rot_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                k = random.randint(1, 3)
                prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(1,2),k=k)
                fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(1,2),k=k)

        prev_frames_tensor = torch.cat([prev_frames_tensor,fut_frames_tensor],dim=0)
       
        # plt.imshow(prev_frames_tensor[0,0,:,:].numpy()+0.5,cmap='gray')
        # plt.savefig('kth_debug.png')
        # exit()
        
        return prev_frames_tensor,fut_frames_tensor

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    filename = root+'/mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNISTPredDataset(Dataset):
    def __init__(self, config, split=True, num_objects=[2],is_train=True):
        super(MovingMNISTPredDataset, self).__init__()
        self.config = config
        self.dataset = None
        self.split = split
        # self.dataset = load_fixed_set(config['dataroot'])
        if split=='train':
            self.mnist = load_mnist(config['dataroot'])
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(config['dataroot'])
            else:
                self.dataset = load_fixed_set(config['dataroot'])
        self.length = int(config['train_length']) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = config['prev_len']
        self.n_frames_output = config['fut_len']
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        # fimages = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)

        images = images.reshape((-1, w, r, w, r))[:length].transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float().to(self.config['device']).permute(0,2,3,1)
        input = torch.from_numpy(input / 255.0).contiguous().float().to(self.config['device']).permute(0,2,3,1)
        input = torch.cat([input,output],dim=0)
        # if self.split == 'train' and self.config['flip_aug']:
        #     flag = random.uniform(0,1)
        #     if flag < 0.5:
        #         input = torch.flip(input,dims=[1])
        #         output = torch.flip(output,dims=[1])
        #     flag = random.uniform(0,1)
        #     if flag < 0.5:
        #         input = torch.flip(input,dims=[2])
        #         output = torch.flip(output,dims=[2])

        # if self.split == 'train' and self.config['rot_aug']:
        #     flag = random.uniform(0,1)
        #     if flag < 0.5:
        #         k = random.randint(1, 3)
        #         input =torch.rot90(input,dims=(1,2),k=k)
        #         output =torch.rot90(output,dims=(1,2),k=k)
            

        
        if self.config['range'] == 255:
            return input * 255., output * 255.
        else:
            return input- 0.5, output- 0.5

    def __len__(self):
        return self.length


if __name__ == "__main__":

    #config = configs.ucf_config
    # config = configs.strpm_ucf_config
    config = configs.davis_config
    
    config['flip_aug'] = False
    config['rot_aug'] = False
    config['sequence_mode'] = 'unique'
    config['range'] = 1
    config['device'] = 'cpu'
    config['flip_aug'] = False
    config['rot_aug'] = False
    config['prev_len'] = 3
    dataset = DavisPredDataset(config,'test')
    import lpips
    lpips = lpips.LPIPS(net='alex')
    static_id = []

    ssim_list = []
    psnr_list = []
    lpips_list = []
    static_num =0
    inf = 0
    for i in tqdm(range(0,len(dataset))):
        prev_frames_tensor, fut_frames_tensor = dataset[i]
        last_frame = prev_frames_tensor[config['prev_len']-1].numpy()+ 0.5
        fut_frame = fut_frames_tensor[0].numpy() + 0.5
        p = psnr(fut_frame,last_frame)
        s = ssim(fut_frame,last_frame,multichannel=True)

        l = lpips(torch.from_numpy(last_frame.copy()).permute(2,0,1).unsqueeze(0),torch.from_numpy(fut_frame.copy()).permute(2,0,1).unsqueeze(0)).item()


        if p > 999999:
            inf += 1
            static_id.append(i)
        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)
        
    print(len(static_id),'/',len(dataset))
    print('inf sample: ',inf)
    static_id = np.asarray(static_id).astype(np.int64)
    print('ssim: ',np.mean(np.asarray(ssim_list)))
    print('psnr: ',np.mean(np.asarray(psnr_list)))
    print('lpips: ',np.mean(np.asarray(lpips_list)))
    result_dict = {'ssim':np.asarray(ssim_list),'psnr':np.asarray(psnr_list),'lpips':np.asarray(lpips_list),'static_id':static_id}
    np.save('davis_test_analysis_3to1.npy',result_dict)





