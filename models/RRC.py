# "Accurate Single Stage Detector Using Recurrent Rolling Convolution" Jimmy Ren et al. (2017) 'RRC'

import torch
import torch.nn as nn
import torch.nn.functonal as F

import os
import numbers
import numpy as np

from vgg import VGG16


# KITTI object detection
class RRC(nn.Module):
    
    img_size = (1272, 375)
    # ((159, 47), (80, 24), (40, 12), (20, 6), (10, 3))
    config = {
        'name': 'RRC-KITTI',
        'image_size' = img_size,
        'grids' = _feature_maps(img_size, start_level=3, n_levels=5, ceil_mode=True),
        'aspect_ratios': ((1/2.,  1,  2), 
                          (1/3.,  1/2.,  1,  2,  3),
                          (1/3.,  1/2.,  1,  2,  3),
                          (1/3.,  1/2.,  1,  2,  3),
                          (1/3.,  1/2.,  1,  2,  3)),
        'sizes': _sizes(img_size, start_level=3, n_levels=5, min_ratio=0.15, max_ratio=0.85),
        'discretize': 2,   # more anchor boxes  # WIP

        'batch_size': 1,
        'init_lr': 0.0005,
        'stepvalues': (30000,),
        'max_iter': 60000,

        'pos_thresh': 0.7,
        'neg_thresh': 0.5,
        'loc_weight': 2.,
    }

    def __init__(self, n_classes=1):
        super().__init__()

        self.n_classes = n_classes
        self.rolling_times = 4
        self.rolling_ratio = 0.075

        self.Base = VGG16()
        self.Extra = nn.Sequential(OrderedDict([
            ('extra1_1', nn.Conv2d(1024, 256, 1)),
            ('extra1_2', nn.Conv2d(256, 256, 3, padding=1, stride=2)),
            ('extra2_1', nn.Conv2d(256, 128, 1)),
            ('extra2_2', nn.Conv2d(128, 256, 3, padding=1, stride=2)),
            ('extra3_1', nn.Conv2d(256, 128, 1)),
            ('extra3_2', nn.Conv2d(128, 256, 3, padding=1, stride=2))]))
        self.pred_layers = ['conv4_3', 'conv7', 'extra1_2', 'extra2_2', 'extra3_2']

        self.L2Norm = nn.ModuleList([L2Norm(512, 20)])
        self.l2norm_layers = ['conv4_3']

        # intermediate layers
        self.Inter = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
            nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1), nn.ReLU(inplace=True))
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential()])
        n_channels = [256, 256, 256, 256, 256]

        # Recurrent Rolling
        self.RollLeft = nn.ModuleList([])
        self.RollRight = nn.ModuleList([])
        self.Roll = nn.ModuleList([])
        for i in range(len(n_channels)):
            n_out = int(n_channels[i] * self.rolling_ratio)
            if i > 0:
                self.RollLeft.append( nn.Sequential(
                    nn.Conv2d(n_channels[i-1], n_out, 1), 
                    nn.ReLU(inplace=True), 
                    nn.MaxPool2d(2, ceil_mode=True)))
            if i < len(n_channels) - 1:
                self.RollRight.append( nn.Sequential(
                    nn.Conv2d(n_channels[i+1], n_out, 1), 
                    nn.Relu(inplace=True), 
                    nn.ConvTranspose2d(n_out, n_out, kernel_size=4, stride=2, padding=1)))

            n_out = n_out * (int(i>0) + int(i<len(n_channels)-1))
            self.Roll.append(nn.Sequential(
                    nn.Conv2d(n_channels[i] + n_out, n_channels[i], 1), 
                    nn.ReLU(inplace=True)))

        # Prediction
        self.Loc = nn.ModuleList([])
        self.Conf = nn.ModuleList([])
        for i in range(len(n_channels)):
            n_boxes = len(self.config['aspect_ratios'][i]) + 1
            self.Loc.append(nn.Conv2d(n_channels[i], n_boxes * 4, 3, padding=1))
            self.Conf.append(nn.Conv2d(n_channels[i], n_boxes * (self.n_classes + 1), 3, padding=1))

    def forward(self, x):
        xs = []
        for name, m in itertools.Chain(self.Base._modules.items(),
                                       self.Extra._modules.items())
            if isinstance(m, nn.Conv2d):
                x = F.relu(m(x), inplace=True)
            else:
                x = m(x)

            if name in self.pred_layers:
                i = self.pred_layers.index(name)
                if name in self.l2norm_layers:
                    y = self.L2Norm[self.l2norm_layers.index(name)](x)
                    xs.append(self.Inter[i](y))
                else:
                    xs.append(self.Inter[i](x))

        return [self._prediction(_) for _ in self.recurrent_rolling(xs)]

    def recurrent_rolling(self, xs):
        num = len(xs)

        ys = [xs]
        for _ in range(self.rolling_times):
            xs = []
            for i in range(num):
                x = []
                if i > 0:
                    x.append(self.RollLeft[i-1](ys[-1][i-1]))
                x.append(ys[-1][i])
                if i < num - 1:
                    x.append(self.RollRight[i](ys[-1][i+1]))
                x = torch.cat(x, dim=1)
                x = self.Roll[i](x)
                xs.append(x)
            ys.append(xs)
        return ys

    def _prediction(self, xs)
        locs = []
        confs = []
        for i, x in enumerate(xs):
            loc = self.Loc[i](x)
            loc = loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.Conf[i](x)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.n_classes + 1)
            confs.append(conf)

        return (torch.cat(locs, dim=1), torch.cat(confs, dim=1))

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





# TODO
# class RRC300(nn.Module):
# class BRRC(nn.Module):



class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(np.ones((1, n_channels, 1, 1))))
        nn.init.constant(self.scale, scale)

    def forward(self, x):   
        x = x * x.pow(2).sum(1, keepdim=True).clamp(min=1e-10).rsqrt()
        return self.scale * x



def _feature_maps(img_size, start_level, n_levels, ceil_mode=False):
    def repeat(f, n):
        if n==0:
            return (lambda x: x)
        return (lambda x: f(repeat(f, n-1)(x)))
    half = lambda x: (int(np.ceil(x / 2.)) if ceil_mode else int(np.floor(x / 2.)))

    if isinstance(img_size, numbers.Number):
        img_size = (img_size, img_size)
    return [(repeat(half, n)(img_size[0]), repeat(half, n)(img_size[1])) for n in range(
        start_level, start_level + n_levels)]


def _sizes(img_size, start_level=3, n_levels=5, min_ratio=0.15, max_ratio=0.85):
    # TODO
    pass