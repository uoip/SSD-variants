# fully conv reduced VGG16
# https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6 , converted to pytorch by amdegroot, https://github.com/amdegroot/ssd.pytorch


import torch
import torch.nn as nn
import numpy as np


class VGG16(nn.Module):
    '''
    input image: BGR format, range [0, 255], then subtract mean
    '''
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, 1)

    def load_pretrained(self, path):
        weights = torch.load(path)

        lookup = {'conv1_1':'0', 'conv1_2':'2', 'conv2_1':'5', 'conv2_2':'7', 
                  'conv3_1':'10', 'conv3_2':'12', 'conv3_3':'14', 
                  'conv4_1':'17', 'conv4_2':'19', 'conv4_3':'21',
                  'conv5_1':'24', 'conv5_2':'26', 'conv5_3':'28',
                  'conv6':'31', 'conv7':'33'}

        model_dict = self.state_dict()
        pretrained_dict = {}
        for name, ind in lookup.items():
            for ext in ['.weight', '.bias']:
                pretrained_dict[name + ext] = weights[ind + ext]
        model_dict.update(pretrained_dict)

        self.load_state_dict(model_dict)