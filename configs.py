import numpy as np

PATH_TO_DATASET = '/home/yiqizhong/project/video_prediction/dataset'

'''
Dataset root
'''
# UCF config

ucf_config = {}
ucf_config['name'] = 'ucf'
ucf_config['dataroot'] = PATH_TO_DATASET + '/ucf_ours/'
ucf_config['in_res'] = (512,512)
ucf_config['prev_len'] = 4
ucf_config['fut_len'] = 1
ucf_config['total_len'] = ucf_config['prev_len'] + ucf_config['fut_len']
ucf_config['eval_list'] = ['psnr','ssim']
ucf_config['shuffle_setting'] = True
ucf_config['downsample_scale'] = np.asarray([2,4,2])
ucf_config['num_probe'] = 9
ucf_config['n_channel'] = 3

#----------------------------------#

strpm_ucf_config = {}
for key in ucf_config.keys():
    strpm_ucf_config[key] = ucf_config[key]
strpm_ucf_config['dataroot'] = PATH_TO_DATASET + '/ucf_strpm/'


#----------------------------------#

kth_config = {}
kth_config['name'] = 'kth'
kth_config['dataroot'] = '/home/yiqizhong/project/video_prediction/dataset/kth/'
kth_config['in_res'] = (128,128)
kth_config['prev_len'] = 10
kth_config['fut_len'] = 20
kth_config['total_len'] = kth_config['prev_len'] + kth_config['fut_len']
kth_config['eval_list'] = ['psnr','ssim']
kth_config['shuffle_setting'] = True
kth_config['downsample_scale'] = np.asarray([2,2,2])
kth_config['n_channel'] = 1

#----------------------------------#

mnist_config = {}
mnist_config['name'] = 'mnist'
mnist_config['dataroot'] = PATH_TO_DATASET + '/moving-mnist/'
mnist_config['in_res'] = (64,64)
mnist_config['prev_len'] = 10
mnist_config['fut_len'] = 10
mnist_config['total_len'] = mnist_config['prev_len'] + mnist_config['fut_len']
mnist_config['eval_list'] = ['psnr','ssim']
mnist_config['shuffle_setting'] = True
mnist_config['downsample_scale'] = np.asarray([2,2,2])
mnist_config['train_length'] = 1e4
mnist_config['n_channel'] = 1 
