# "Residual Features and Unified Prediction Network for Single Stage Detection" Kyoungmin Lee et al. (2017) 'RUN'

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 

import numpy as np
import os 
import itertools
from collections import OrderedDict

from vgg import VGG16



class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=20):
        super(L2Norm,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out



class RUN300(nn.Module):
    
    config = {
        'name': 'RUN300-VGG16',
        'image_size': 300,
        'grids': (38, 19, 10, 5, 3, 1),
        'aspect_ratios': (1/3.,  1/2.,  1,  2,  3),  # 4 or 6
        'steps': [s / 300. for s in [8, 16, 32, 64, 100, 300]],
        'sizes': [s / 300. for s in (30, 60, 111, 162, 213, 264, 315)],  
        #'sizes': [s / 300. for s in (30, 60, 104, 157, 210, 264, 315)],
    } 

    def __init__(self, n_classes):
        super(RUN300, self).__init__()
        self.n_classes = n_classes

        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([
            ('extra1_1', nn.Conv2d(1024, 256, 1)),
            ('extra1_2', nn.Conv2d(256, 512, 3, padding=1, stride=2)),
            ('extra2_1', nn.Conv2d(512, 128, 1)),
            ('extra2_2', nn.Conv2d(128, 256, 3, padding=1, stride=2)),
            ('extra3_1', nn.Conv2d(256, 128, 1)),
            ('extra3_2', nn.Conv2d(128, 256, 3)),
            ('extra4_1', nn.Conv2d(256, 128, 1)),
            ('extra4_2', nn.Conv2d(128, 256, 3))]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2', 'extra3_2','extra4_2']
        n_channels = [512, 1024, 512, 256, 256, 256]

        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.l2norm_layers = ['conv4_3']

        # Multi-Way Residual Blocks
        self.ResBlocks = nn.ModuleList()
        for i in range(len(n_channels) - 1):
            self.ResBlocks.append(
                ThreeWay(n_channels[i], n_channels[i+1], self.config['grids'][i], self.config['grids'][i+1], 
                         out_channels=256))
        self.ResBlocks.append(TwoWay(n_channels[-1], out_channels=256))

        # Unified Prediction Module
        n_boxes = len(self.config['aspect_ratios']) + 1
        #self.Loc  = nn.Conv2d(256, n_boxes * 4, 3, padding=1)
        #self.Conf = nn.Conv2d(256, n_boxes * (self.n_classes+1), 3, padding=1)
        self.Loc = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_boxes * 4, 3, padding=1))
        self.Conf = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_boxes * (self.n_classes+1), 3, padding=1))
        

    def forward(self, x):
        xs = []
        for name, m in itertools.chain(self.Base._modules.items(), 
                                       self.Extra._modules.items()):
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)

            if name in self.pred_layers:
                if name in self.l2norm_layers:
                    i = self.l2norm_layers.index(name)
                    xs.append(self.L2Norm[i](x))
                else:
                    xs.append(x)

        return self._prediction(self.multiway(xs))

    def multiway(self, xs):
        ys = []
        for i in range(len(xs)):
            block = self.ResBlocks[i]
            if isinstance(block, ThreeWay):
                y = block(xs[i], xs[i+1])
                ys.append(y)
            elif isinstance(block, TwoWay):
                y = block(xs[i])
                ys.append(y)
        return ys

    def _prediction(self, ys):
        locs = []
        confs = []
        for y in ys:
            loc = self.Loc(y)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.Conf(y)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)


    def init_parameters(self, backbone=None):
        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.apply(weights_init)

        if backbone is not None and os.path.isfile(backbone):
            self.Base.load_pretrained(backbone)





# pre-activation
class TwoWay(nn.Module):
    def __init__(self, in_channels, out_channels=256, bypass=False):
        super().__init__()

        if bypass and in_channels == out_channels:
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = conv_relu(in_channels, out_channels, 1)

        self.branch2 = nn.Sequential(
            conv_relu(in_channels, out_channels // 2, 1),
            conv_relu(out_channels // 2, out_channels // 2, 3, padding=1),
            conv_relu(out_channels // 2, out_channels, 1))

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class ThreeWay(TwoWay):
    def __init__(self, in_channels1, in_channels2, in_size1, in_size2, out_channels=256):
        super().__init__(in_channels1, out_channels)

        self.branch3 = nn.Sequential(
            conv_relu(in_channels2, out_channels // 2, 3, padding=1),
            deconv_relu(out_channels // 2, out_channels // 2, in_size2, in_size1),
            conv_relu(out_channels // 2, out_channels, 1))

    def forward(self, x1, x2):
        return self.branch1(x1) + self.branch2(x1) + self.branch3(x2)

        


def deconv_relu(in_channels, out_channels, in_size, out_size):
    # TODO: handle case when size is tuple
    if out_size == 2 * in_size:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    elif out_size == 2 * in_size - 1:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    elif out_size == 2 * in_size + 1:
        dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
    else:
        raise ValueError('invalid size')
    return nn.Sequential(dconv, nn.ReLU(inplace=True))


def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True))


'''
def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True))


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

def bn_relu_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
'''