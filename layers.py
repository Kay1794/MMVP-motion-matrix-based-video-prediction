import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import sim_matrix_postprocess
import torchvision
import time
#from simvp_model import Inception

'''
RRDB 

code is modified from https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py ESRGAN/RRDBNet_arch.py 

'''

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_4C(nn.Module):
    def __init__(self, nf=64, gc = 32,bias=True):
        super(ResidualDenseBlock_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf):
        super(RRDB, self).__init__()
        gc = nf // 2
        self.RDB1 = ResidualDenseBlock_4C(nf, gc)
        self.RDB2 = ResidualDenseBlock_4C(nf, gc)
        self.RDB3 = ResidualDenseBlock_4C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBEncoder(nn.Module):
    def __init__(self, config):
        super(RRDBEncoder, self).__init__()

        base_channel = config['base_channel']
        bilinear = config['bilinear']
        scale = config['downsample_scale']
        self.scale_num = len(scale)
        n_channel = config['n_channel'] * 4 if config['shuffle_setting'] else config['n_channel']
        self.config = config
        self.inconv = nn.Conv2d(n_channel, base_channel, 3, 1, 1)
        layers = []
        for i in range(config['rrdb_encoder_num']):
            layers.append(RRDB(base_channel))
        self.block_rrdb = nn.Sequential(*layers)

        pre_downsample_block_list = []
        
        for i in range(self.scale_num-2):
            pre_downsample_block_list.append(ResBlock(config,base_channel * (2**(i)),base_channel* (2**((i+1))),downsample=True,factor=scale[i]))
        self.pre_downsample_block = nn.ModuleList(pre_downsample_block_list)
        
        self.downsample_high = ResBlock(config,base_channel*(2**((self.scale_num-2))),base_channel*(2**((self.scale_num-1))),downsample=True,factor=scale[-2])
        self.downsample_low = ResBlock(config,base_channel*(2**((self.scale_num-1))),base_channel*(2**((self.scale_num))),downsample=True,factor=scale[-1])


    def forward(self, x, save_all=False):
        in_feat = []
        x = self.inconv(x)
        x1 = self.block_rrdb(x)
        x2 = x1
        in_feat.append(x2.clone())
        for i in range(self.scale_num-2):
            x2 = self.pre_downsample_block[i](x2) 
            in_feat.append(x2.clone())
        x_high = self.downsample_high(x2)
        in_feat.append(x_high.clone())
        x_low = self.downsample_low(x_high)
        in_feat.append(x_low.clone())
        if self.config['scale_in_use'] == '3':
            for i in range(len(in_feat)-3):
                in_feat[i] = None
        elif self.config['scale_in_use'] == '2':
            for i in range(len(in_feat)-2):
                in_feat[i] = None
        return in_feat



class RRDBDecoder(nn.Module):
    def __init__(self, config):
        super(RRDBDecoder, self).__init__()

        base_channel = config['base_channel']
        bilinear = config['bilinear']
        scale = config['downsample_scale']
        self.scale_num = len(scale)
        out_channel = config['n_channel'] * 4 if config['shuffle_setting'] else config['n_channel']

        factor = 2 if bilinear else 1
        
        self.upsample_block_low2high = Up(config,base_channel*(2**((self.scale_num))), base_channel*(2**((self.scale_num-1))), bilinear, scale=scale[-1])

        upsample_block_list = []
        for i in range(self.scale_num-2,-1,-1):
            skip=False if (((i<self.scale_num-1) and (config['scale_in_use'] == '2')) or ((i<self.scale_num-2) and (config['scale_in_use'] == '3'))) else True
            upsample_block_list.append(Up(config,base_channel * (2**((i+1))), base_channel * (2**(i)), bilinear,scale=scale[i],skip=skip))
        self.upsample_block =  nn.ModuleList(upsample_block_list)

        layers = []
        for i in range(config['rrdb_decoder_num']):
            layers.append(RRDB(config['base_channel']))
        self.rrdb_block= nn.Sequential(*layers)

        self.outc = OutConv(base_channel, out_channel)

    def forward(self, in_feat):

        x = self.upsample_block_low2high(in_feat[-1], in_feat[-2])
        for i in range(self.scale_num-1):
            x = self.upsample_block[i](x,in_feat[-i-3])

        x = self.rrdb_block(x)
        logits = self.outc(x)
        return logits

#---------------------------------#


class ResBlock(nn.Module):
    def __init__(self, config,in_channels, out_channels, downsample=False,upsample=False,skip=False,factor=2,motion=False):
        super().__init__()
        self.upsample = upsample
        self.config = config
        self.maxpool= None
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels) 
            ) if config['use_bn'] else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if factor == 4:
                self.maxpool = nn.MaxPool2d(2)
            
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)) \
            if (config['use_bn'] or motion) else nn.Sequential(nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if config['use_bn']:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input))) if self.config['use_bn'] else nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.bn2(self.conv2(input))) if self.config['use_bn'] else nn.ReLU()(self.conv2(input))
        #if not self.upsample:
        input = input + shortcut
        if not self.maxpool is None:
            input = self.maxpool(input)
        return nn.LeakyReLU()(input)



class ImageEnhancer(nn.Module):
    def __init__(self, config,n_channels=3, base_channel=16,scale=[2,2,2]):
        super(ImageEnhancer, self).__init__()
        self.n_channels = n_channels
        layers = [nn.Conv2d(n_channels*2 if (config['res_cat_img']) else n_channels, base_channel,3,1,1)]
        for i in range(config['rrdb_enhance_num']):
            layers.append(RRDB(base_channel))
        self.model = nn.Sequential(*layers)


        self.outconv = OutConv(base_channel,n_channels)
    


    def forward(self, x,):
        feat = self.model(x)
        out = self.outconv(feat)

        return out


  

class ConvLayer(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, config,in_channels, out_channels, mid_channels=None,bn=True,motion=False,dilation=1):
        super().__init__()
        self.config = config
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )  if ((config['use_bn']) or (config['motion_use_bn'] and motion)) else  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False,dilation=dilation),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, config,in_channels, out_channels,scale=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale),
            ConvLayer(config,in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,config, in_channels, out_channels, bilinear=True, skip=True,scale=2,bn=True,motion=False):
        super().__init__()
        factor = scale
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if skip:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(config,in_channels, out_channels,bn=bn)

            else:
                self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
                self.conv = ConvLayer(config,in_channels, out_channels)

        else:
            if skip:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
                self.conv = ConvLayer(config,out_channels*2, out_channels,bn=bn,motion=motion)
            else:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
                self.conv = ConvLayer(config,out_channels, out_channels,bn=bn,motion=motion)

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        if x2 is None:
            return self.conv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)


from torch import nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Conv3D(nn.Module):
    def __init__(self, config,in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.config = config
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.config['motion_use_bn']:
            self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.leaky_relu(self.bn3d(self.conv3d(x))) if self.config['motion_use_bn'] else  F.leaky_relu(self.conv3d(x))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


class MatrixPredictor3DConv(nn.Module):
    def __init__(self, config,hidden_len=64):
        super(MatrixPredictor3DConv, self).__init__()
        self.unet_base = hidden_len #64
        self.hidden_len = hidden_len #64
        self.config = config
        self.conv_pre_1 = nn.Conv2d(hidden_len,hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)      

        self.conv3d_1 = Conv3D(config,self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = Conv3D(config,self.unet_base*2, self.unet_base*2, kernel_size=(3  , 3, 3), stride=1, padding=(0, 1, 1))

        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        
        if config['motion_use_bn']:
            self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
            self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
            self.bn1_1 = nn.BatchNorm2d(self.unet_base)
            self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
            self.bn3_1 = nn.BatchNorm2d(self.unet_base)
            self.bn4_1 = nn.BatchNorm2d(self.hidden_len)
            
        
    def forward(self,x):
        # x [B,T,C,32,32]
        # out: [B,C,32,32]
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv_pre_1(x))
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv_pre_2(x))
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv1_1(x))
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1) #  (batch, seq, c, h, w), 1st temporal conv
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1)))  if self.config['motion_use_bn'] else F.leaky_relu(self.conv2_1(x_1)) # (batch * seq, c, h // 2, w // 2)
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2) # (batch, seq=1, c, h // 2, w // 2), 2nd temporal conv
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h//2, w//2), seq = 1
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)) # (batch, seq, c, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous() # (batch, c, seq, h, w)                                           
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None)) # (batch, c, 1, h, w)
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous() # (batch, 1, c, h, w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous() # (batch*1, c, h, w)
        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1)))) if self.config['motion_use_bn'] else\
        F.leaky_relu(self.conv3_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1)))
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3)) # (batch, seq, 1, h, w)
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, scale_factor=(2, 2))))) if self.config['motion_use_bn'] else\
        F.leaky_relu(self.conv4_1(F.interpolate(x_3, scale_factor=(2, 2))))
        
        return x


class SimpleMatrixPredictor3DConv_direct(nn.Module):
    def __init__(self, config,hidden_len=64,image_pred=False):
        super(SimpleMatrixPredictor3DConv_direct, self).__init__()
        self.unet_base = hidden_len #64
        self.hidden_len = hidden_len #64
        self.config = config
        self.conv_pre_1 = nn.Conv2d(hidden_len,hidden_len, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(hidden_len, hidden_len, kernel_size=3, stride=1, padding=1)
        self.fut_len = 1 if config['long_term'] else config['fut_len']

        self.conv3d_1 = Conv3D(config,self.unet_base, self.unet_base, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        
        if self.fut_len > 1 :
            self.temporal_layer = Conv3D(config,self.unet_base*2, self.unet_base*2, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        else:
            self.temporal_layer = nn.Sequential(
            nn.Conv2d(self.unet_base *2, self.unet_base * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU())

        input_len = config['prev_len'] if image_pred else (config['prev_len']-1)
        self.conv_translate = nn.Sequential(
            nn.Conv2d(self.unet_base * input_len , self.unet_base * self.fut_len, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU())
        
        self.conv1_1 = nn.Conv2d(hidden_len, self.unet_base, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.unet_base, self.unet_base * 2, kernel_size=3, stride=2, padding=1)
        
        self.conv3_1 = nn.Conv2d(self.unet_base * 3, self.unet_base, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(self.unet_base, self.hidden_len, kernel_size=3, stride=1, padding=1)
        
        if config['motion_use_bn']:
            self.bn_pre_1 = nn.BatchNorm2d(hidden_len)
            self.bn_pre_2 = nn.BatchNorm2d(hidden_len)
            self.bn1_1 = nn.BatchNorm2d(self.unet_base)
            self.bn2_1 = nn.BatchNorm2d(self.unet_base * 2)
            self.bn3_1 = nn.BatchNorm2d(self.unet_base)
            self.bn4_1 = nn.BatchNorm2d(self.hidden_len)
            self.bn_translate = nn.BatchNorm2d(self.unet_base * self.fut_len)
            
        
    def forward(self,x):
        # x [B,T,C,32,32]
        # out: [B,C,32,32]
        batch, seq, z, h, w = x.size()
        x = x.reshape(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.leaky_relu(self.bn_pre_1(self.conv_pre_1(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv_pre_1(x))
        x = F.leaky_relu(self.bn_pre_2(self.conv_pre_2(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv_pre_2(x))
        x_1 = F.leaky_relu(self.bn1_1(self.conv1_1(x))) if self.config['motion_use_bn'] else F.leaky_relu(self.conv1_1(x))
        
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        
        x_1 = self.conv3d_1(x_1) #  (batch, seq, c, h, w), 1st temporal conv
        batch, seq, c, h, w = x_1.shape
        x_tmp = x_1.reshape(batch,-1,h,w)
        x_tmp = self.bn_translate(self.conv_translate(x_tmp)) if self.config['motion_use_bn'] else self.conv_translate(x_tmp)
        x_1 = x_tmp.reshape(batch,self.fut_len,c,h,w)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)
        x_2 = F.leaky_relu(self.bn2_1(self.conv2_1(x_1))) # (batch * seq, c, h // 2, w // 2)
        if self.fut_len > 1:
            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_2 = self.temporal_layer(x_2) # (batch, seq=10, c, h // 2, w // 2)
        
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h//2, w//2), seq = 1
        else:
            x_2 = self.temporal_layer(x_2) # (batch * seq,c, h // 2, w // 2)

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)) # (batch, seq, c, h, w)
        
        x_1 = x_1.reshape(-1, x_1.size(2), x_1.size(3), x_1.size(4))


        x_3 = F.leaky_relu(self.bn3_1(self.conv3_1(torch.cat((F.interpolate(x_2, size=x_1.shape[2:]), x_1), dim=1))))
        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3)) # (batch, seq, 1, h, w)
        x = F.leaky_relu(self.bn4_1(self.conv4_1(F.interpolate(x_3, size = x.shape[3:])))) if self.config['motion_use_bn'] else\
        F.leaky_relu(self.conv4_1(F.interpolate(x_3, size = x.shape[3:])))
        
        return x

class PredictModel(nn.Module):
    def __init__(self,config=None,hidden_len=32,mx_h=32,mx_w=32):
        super(PredictModel, self).__init__()
        self.mx_h = mx_h
        self.mx_w = mx_w
        self.config =config
        self.hidden_len = hidden_len
        self.fut_len = 1 if config['long_term'] else config['fut_len']
        self.in_conv = nn.Conv2d( 1, hidden_len, kernel_size=3, padding=1, bias=False)
        if config['filter_block']:
            self.in_conv = nn.Conv2d(1, hidden_len, kernel_size=3, padding=1, bias=False)
        self.fuse_conv = nn.Conv2d(hidden_len*2, hidden_len, kernel_size=3, padding=1, bias=False)
        if config['use_direct_predictor']:
            self.predictor = SimpleMatrixPredictor3DConv_direct(config,hidden_len)
        else:
            self.predictor = MatrixPredictor3DConv(config,hidden_len)
        self.out_conv = nn.Conv2d(hidden_len, 1, kernel_size=3, padding=1, bias=False)
        # self.out_conv = DeformConv(hidden_len, 1, kernel_size=3, padding=1,device=config['device'])
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def res_interpolate(self,in_tensor,template_tensor):
        '''
        in_tensor: batch,c,h'w',H'W'
        tempolate_tensor: batch,c,hw,HW
        out_tensor: batch,c,hw,HW
        '''
        out_tensor = F.interpolate(in_tensor,template_tensor.shape[-2:]) # (BThw,target_h,target_w)

        return out_tensor

    def forward(self,matrix_seq,softmax=False,res=None):

        B,T,hw,window_size = matrix_seq.size()

        c = self.hidden_len
        matrix_seq = matrix_seq.reshape(-1,hw,self.mx_h,self.mx_w) # (BT,hw,hw)
        matrix_seq = matrix_seq.reshape(B*T*hw,self.mx_h,self.mx_w).unsqueeze(1) # (BThw,1,h,w)

        x = self.in_conv(matrix_seq)
        x = x.reshape(B,T,hw,-1,self.mx_h,self.mx_w)
        x = x.permute(0,2,1,3,4,5).reshape(B*hw,T,-1,self.mx_h,self.mx_w)
        emb = self.predictor(x)

        emb = emb.reshape(B*hw*self.fut_len,-1,self.mx_h,self.mx_w)
        res_emb = emb.clone()
        if not res is None:
            template = emb.clone().reshape(B,hw,emb.shape[1],-1).permute(0,2,1,3)
            in_tensor = res.clone().reshape(B,hw//4,emb.shape[1],-1).permute(0,2,1,3)
            
            res_tensor = self.res_interpolate(in_tensor,template).permute(0,2,1,3).reshape(emb.shape)
            
            emb = self.fuse_conv(torch.cat([emb,res_tensor],dim=1))

        out = self.out_conv(emb) #(Bhwt,16,h//4,w//4)
        
        out = out.reshape(B,hw,-1,self.mx_h,self.mx_w)
        out = out.permute(0,2,1,3,4)
        out = out.reshape(B,-1,hw,window_size)
        
        if softmax:
            out = out.view(B,out.shape[1],-1)
            out = self.softmax(out)
            out = out.reshape(B,-1,hw,window_size)

        return out,res_emb

