# "DSOD: Learning Deeply Supervised Object Detectors from Scratch" Zhiqiang Shen et al. ICCV17

import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
from collections import OrderedDict



class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(np.ones((1, n_channels, 1, 1))))
        nn.init.constant(self.scale, scale)

    def forward(self, x):   
        x = x * x.pow(2).sum(1, keepdim=True).clamp(min=1e-10).rsqrt()
        return self.scale * x



class DSOD300(nn.Module):

    config = {
        'name': 'DSOD300-64-192-48-1',
        'image_size': 300,
        'grids': (38, 19, 10, 5, 3, 1),
        'aspect_ratios': ((1/2.,  1,  2),   
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/3.,  1/2.,  1,  2,  3), 
                          (1/2.,  1,  2),
                          (1/2.,  1,  2)),
        #'sizes': [s / 300. for s in (30, 60, 101, 152, 206, 264, 315)],
        'steps': [s / 300. for s in [8, 16, 32, 64, 100, 300]],
        'sizes': [s / 300. for s in [30, 60, 111, 162, 213, 264, 315]],

        #'n_gpus': 8,
        #'batch_size': 128,
        #'init_lr': 0.1,
        #'stepvalues': (20000, 40000, 60000, 80000),
        #'max_iter': 100000,
    }

    def __init__(self, n_classes, growth_rate=48):
        super(DSOD300, self).__init__()
        self.n_classes = n_classes

        depth = [6, 8, 8, 8]
        channels = [128 + growth_rate * _ for _ in np.cumsum(depth)]   
        # growth_rate 48: 416, 800, 1184, 1568

        # backbone
        self.Stem = nn.Sequential(
            conv_bn_relu(3, 64, 3, stride=2, padding=1),
            conv_bn_relu(64, 64, 3, padding=1),
            conv_bn_relu(64, 128, 3, padding=1),
            nn.MaxPool2d(2, ceil_mode=True))

        self.Block12 = nn.Sequential(
            DenseBlock(128, depth[0], growth_rate),
            Transition(channels[0], channels[0], pool=True, ceil_mode=True),  # 75 -> 38
            DenseBlock(channels[0], depth[1], growth_rate),
            Transition(channels[1], channels[1]))

        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)   # 38 -> 19
        self.conv2 = bn_relu_conv(channels[1], 256, 1)

        self.Block34 = nn.Sequential(
            DenseBlock(channels[1], depth[2], growth_rate),
            Transition(channels[2], channels[2]),
            DenseBlock(channels[2], depth[3], growth_rate),
            Transition(channels[3], 256))

        # extra layers
        self.Extra = nn.ModuleList([
            LHRH(512, 512, ceil_mode=True),            # 19 -> 10
            LHRH(512, 256, ceil_mode=True),            # 10 -> 5
            LHRH(256, 256, ceil_mode=True),            #  5 -> 3
            LHRH(256, 256)])                           #  3 -> 1
        n_channels = [channels[1], 512, 512, 256, 256, 256]

        # prediction layers
        self.L2Norm = nn.ModuleList()
        self.Loc = nn.ModuleList()
        self.Conf = nn.ModuleList()
        for i, ar in enumerate(self.config['aspect_ratios']):
            n = len(ar) + 1
            self.L2Norm.append(L2Norm(n_channels[i], 20))
            self.Loc.append(nn.Conv2d(n_channels[i], n * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n * (self.n_classes + 1), 3, padding=1))

        # weights initialization
        self.apply(self.weights_init)

    def forward(self, x):
        xs = []
        x = self.Stem(x)
        x = self.Block12(x)
        xs.append(x)

        x = self.pool2(x)
        x2 = self.conv2(x)
        x = self.Block34(x)
        x = torch.cat([x2, x], dim=1)
        xs.append(x)

        for m in self.Extra:
            x = m(x)
            xs.append(x)

        return self._prediction(xs)

    def _prediction(self, xs):
        locs = []
        confs = []
        for i, x in enumerate(xs):
            x = self.L2Norm[i](x)

            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def init_parameters(self, x):
        pass
            




def bn_relu_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



class DenseBlock(nn.Module):
    def __init__(self, in_channels, block_depth, growth_rate=48):
        super(DenseBlock, self).__init__()

        class DenseLayer(nn.Module):
            def __init__(self, in_channels, growth_rate, widen=1, dropout=0.):
                super(DenseLayer, self).__init__()

                self.conv1 = bn_relu_conv(in_channels, growth_rate * widen, 1)
                self.conv2 = bn_relu_conv(growth_rate * widen, growth_rate, 3, padding=1)
                self.dropout = dropout

            def forward(self, x):
                out = self.conv1(x)
                out = self.conv2(out)
                if self.dropout > 0:
                    out = F.dropout(out, p=self.dropout, training=self.training)
                return torch.cat([x, out], 1)

        layers = []
        for i in range(block_depth):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, ceil_mode=False, dropout=0.):
        super(Transition, self).__init__()

        self.conv = bn_relu_conv(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2, ceil_mode=ceil_mode) if pool else nn.Sequential()
        self.dropout = dropout

    def forward(self, x):
        out = self.conv(x)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return self.pool(out)


# Learning Half and Reusing Half
class LHRH(nn.Module):
    def __init__(self, in_channels, out_channels, widen=1, dropout=0., ceil_mode=False):
        super(LHRH, self).__init__()

        self.conv1_1 = bn_relu_conv(in_channels, int(out_channels / 2 * widen), 1)
        self.conv1_2 = bn_relu_conv(int(out_channels / 2 * widen), out_channels // 2, 3, 
                                    padding=1 * ceil_mode, stride=2)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=ceil_mode)
        self.conv2 = bn_relu_conv(in_channels, out_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x):
        out1 = self.conv1_2(self.conv1_1(x))
        out2 = self.conv2(self.pool2(x))
        if self.dropout > 0:
            out1 = F.dropout(out1, p=self.dropout, training=self.training)
            out2 = F.dropout(out2, p=self.dropout, training=self.training)
        return torch.cat([out1, out2], 1)