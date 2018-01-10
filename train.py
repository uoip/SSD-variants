import os
import time
import argparse
import random
import numpy as np 
from numpy.random import RandomState
import cv2

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pascal_voc import VOCDetection, VOC, Viz
from transforms import *
from evaluation import eval_voc_detection

import sys
sys.path.append('./models')
from SSD import SSD300

from loss import MultiBoxLoss
from multibox import MultiBox


from tensorboardX import SummaryWriter
summary = SummaryWriter()

# Setup
parser = argparse.ArgumentParser(description='PyTorch SSD variants implementation')
parser.add_argument('--checkpoint', help='resume from checkpoint', default='')
parser.add_argument('--voc_root', help='PASCAL VOC dataset root path', default='')
parser.add_argument('--batch_size', type=int, help='input data batch size', default=32)
parser.add_argument('--lr', '--learning_rate', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--start_iter', type=int, help='start iteration', default=0)
parser.add_argument('--backbone', help='pretrained backbone net weights file', default='vgg16_reducedfc.pth')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--demo', action='store_true', help='show detection result')
parser.add_argument('--seed', type=int, help='random seed', default=233)
parser.add_argument('--threads', type=int, help='number of data loader workers.', default=4)

opt = parser.parse_args()
print('argparser:', opt)




# random seed
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    assert torch.cuda.is_available(), 'No GPU found, please run without --cuda'
    torch.cuda.manual_seed_all(opt.seed)


# model
model = SSD300(VOC.N_CLASSES)
cfg = model.config

if opt.checkpoint:
    model.load_state_dict(torch.load(opt.checkpoint))
else:
    model.init_parameters(opt.backbone)

encoder = MultiBox(cfg)
criterion = MultiBoxLoss()

# cuda
if opt.cuda:
    model.cuda()
    criterion.cuda()
    cudnn.benchmark = True

# optimizer
optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# learning rate / iterations
init_lr = cfg.get('init_lr', 1e-3)
stepvalues = cfg.get('stepvalues', (80000, 100000))
max_iter = cfg.get('max_iter', 120000)

def adjust_learning_rate(optimizer, stage):
    lr = init_lr * (0.1 ** stage)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def warm_up(iteration):
# 	warmup_steps = [0, 300, 600, 1000]
# 	if iteration in warmup_steps:
# 		i = warmup_steps.index(iteration)
# 		lr = 10 ** np.linspace(-6, np.log10(init_lr), len(warmup_steps))[i]
# 		for param_group in optimizer.param_groups:
# 			param_group['lr'] = lr


def learning_rate_schedule(iteration):
    if iteration in stepvalues:
        adjust_learning_rate(optimizer, stepvalues.index(iteration) + 1)




def train():   
    model.train()
    PRNG = RandomState(opt.seed)

    # augmentation / transforms
    transform = Compose([
            [ColorJitter(prob=0.5)],  # or write [ColorJitter(), None]
            BoxesToCoords(),
            Expand((1, 4), prob=0.5),
            ObjectRandomCrop(),
            HorizontalFlip(),
            Resize(300),
            CoordsToBoxes(),
            [SubtractMean(mean=VOC.MEAN)],
            [RGB2BGR()],
            [ToTensor()],
            ], PRNG, mode=None, fillval=VOC.MEAN)
    target_transform = encoder.encode

    dataset = VOCDetection(
            root=opt.voc_root, 
            image_set=[('2007', 'trainval'), ('2012', 'trainval'),],
            keep_difficult=True,
            transform=transform,
            target_transform=target_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, 
            num_workers=opt.threads, pin_memory=True)

    iteration = opt.start_iter
    while True:
        for input, loc, label in dataloader:
            learning_rate_schedule(iteration)

            input, loc, label = Variable(input), Variable(loc), Variable(label)
            if opt.cuda:
                input, loc, label = input.cuda(), loc.cuda(), label.cuda()

            xloc, xconf = model(input)
            loc_loss, conf_loss = criterion(xloc, xconf, loc, label)
            loss = loc_loss + conf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                summary.add_scalar('loss/loc_loss', loc_loss.data[0], iteration)
                summary.add_scalar('loss/conf_loss', conf_loss.data[0], iteration)
                summary.add_scalars('loss/loss', {"loc_loss": loc_loss.data[0],
                                                  "conf_loss": conf_loss.data[0],
                                                  "loss": loss.data[0]}, iteration)

            if iteration % 5000 == 0 and iteration != opt.start_iter:
                print('Save state, iter: {}, Loss:{}'.format(iteration, loss.data[0]))
                if not os.path.isdir('weights'):
                        os.mkdir('weights')
                torch.save(model.state_dict(), 'weights/{}_0712_{}.pth'.format(cfg.get('name', 'SSD'), iteration))
            
            iteration += 1
            if iteration > max_iter:
                return 0

            



def test():
    PRNG = RandomState()

    dataset = VOCDetection(
            root=opt.voc_root, 
            image_set=[('2007', 'test')],
            transform=Compose([
                BoxesToCoords(),
                Resize(300),
                CoordsToBoxes(),
                [SubtractMean(mean=VOC.MEAN)],
                [RGB2BGR()],
                [ToTensor()]]),
            target_transform=None)
    print(len(dataset))

    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = [], [], [], [], []
    for i in range(len(dataset)):
        img, loc, label = dataset[i]

        gt_bboxes.append(loc)
        gt_labels.append(label)

        input = Variable(img.unsqueeze(0), volatile=True)
        if opt.cuda:
            input = input.cuda()

        xloc, xconf = model(input)
        xloc = xloc.data.cpu().numpy()[0]
        xconf = xconf.data.cpu().numpy()[0]

        boxes, labels, scores = encoder.decode(xloc, xconf, nms_thresh=0.5, conf_thresh=0.01)

        pred_bboxes.append(boxes)
        pred_labels.append(labels)
        pred_scores.append(scores)

    print(eval_voc_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=0.5, use_07_metric=True))

        

def demo():
    PRNG = RandomState()
    voc_vis = Viz()

    dataset = VOCDetection(
            root=opt.voc_root, 
            image_set=[('2007', 'test')],
            transform=Compose([
                BoxesToCoords(),
                Resize(300),
                CoordsToBoxes(),
                [SubtractMean(mean=VOC.MEAN)],
                [RGB2BGR()],
                [ToTensor()]]),
            target_transform=None)
    print(len(dataset))

    i = PRNG.choice(len(dataset))
    for _ in range(1000):
        img, _, _ = dataset[i]

        input = Variable(img.unsqueeze(0), volatile=True)
        if opt.cuda:
            input = input.cuda()

        xloc, xconf = model(input)

        imgs = input.data.cpu().numpy().transpose(0, 2, 3, 1)
        xloc = xloc.data.cpu().numpy()
        xconf = xconf.data.cpu().numpy()
        
        for img, loc, conf in zip(imgs, xloc, xconf):
            #print(img.mean(axis=(0,1)))
            img = ((img[:, :, ::-1] + VOC.MEAN)).astype('uint8')
            boxes, labels, scores = encoder.decode(loc, conf, conf_thresh=0.5)

            img = voc_vis.draw_bbox(img, boxes, labels, True)
            cv2.imshow('0', img[:, :, ::-1])
            c = cv2.waitKey(0)
            if c == 27 or c == ord('q'):   # ESC / 'q'
                return
            else:
                i = PRNG.choice(len(dataset))





if __name__ == '__main__':
    if opt.test:
        test()
    elif opt.demo:
        demo()
    else:
        train()