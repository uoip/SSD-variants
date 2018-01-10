import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def log_sum_exp(x, dim, keepdim=False):
    x_max = x.max(dim=dim, keepdim=True)[0]

    if keepdim:
        return (x - x_max).exp().sum(dim=dim, keepdim=True).log() + x_max
    else:
        return (x - x_max).exp().sum(dim=dim).log() + x_max.squeeze(dim)


def _softmax_cross_entropy_with_logits(x, t):
    assert x.size()[:-1] == t.size()
    xt = torch.gather(x, -1, t.long().unsqueeze(-1))
    return log_sum_exp(x, dim=-1, keepdim=False) - xt.squeeze(-1)
    


class MultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _hard_negative_mining(self, loss, pos, neg, k):
        loss = loss.detach()
        rank = (loss * (-1 * neg.float())).sort(dim=1)[1].sort(dim=1)[1]
        hard_neg = rank < (pos.long().sum(dim=1, keepdim=True) * k)
        return hard_neg

    def forward(self, xloc, xconf, loc, label, k=3):   # xconf is logits
        pos = label > 0
        neg = label == 0
        label = label.clamp(min=0)

        pos_idx = pos.unsqueeze(-1).expand_as(xloc)
        loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx].view(-1, 4), 
                                    size_average=False) 
        
        conf_loss = _softmax_cross_entropy_with_logits(xconf, label)
        hard_neg = self._hard_negative_mining(conf_loss, pos, neg, k)
        conf_loss = conf_loss * (pos + hard_neg).gt(0).float()
        conf_loss = conf_loss.sum()

        N = pos.data.float().sum() + 1e-3#.clamp(min=1e-3)
        return loc_loss / N, conf_loss / N





def _softmax_focal_loss(x, t, gamma=2):
    assert x.size()[:-1] == t.size()
    logp = torch.gather(F.log_softmax(x), -1, t.long().unsqueeze(-1))
    FL = - (1 - logp.exp()).pow(gamma) * logp
    return FL.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.count = 0
        self.multiloss = MultiBoxLoss()

    def forward(self, xloc, xconf, loc, label):
        pos = label > 0
        neg = label == 0

        pos_idx = pos.unsqueeze(-1).expand_as(xloc)
        loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx].view(-1, 4), 
                                    size_average=False) 

        pos_idx = pos.unsqueeze(-1).expand_as(xconf)
        pos_conf_loss = _softmax_focal_loss(xconf[pos_idx].view(-1, xconf.size(-1)), label[pos])
        neg_idx = neg.unsqueeze(-1).expand_as(xconf)
        neg_conf_loss = _softmax_focal_loss(xconf[neg_idx].view(-1, xconf.size(-1)), label[neg])

        conf_loss = self.alpha * pos_conf_loss + (1 - self.alpha) * neg_conf_loss

        self.count += 1
        if self.count % 1000 == 0:
            print('pos loss, neg loss', pos_conf_loss.data, neg_conf_loss.data)
            print('multiloss', self.multiloss(xloc, xconf, loc, label, 3))
        
        N = pos.float().sum().clamp(min=1e-3)
        return loc_loss / N, conf_loss / N




# class SigmoidFocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, scale=4.):
#         super().__init__()
#         self.alpha = alpha
#         self.scale = scale
#         self.onehot = None

#     def forward(self, xloc, xconf, loc, label):
#         pos = label > 0
#         neg = label == 0

#         # loc
#         pos_idx = pos.unsqueeze(-1).expand_as(xloc)
#         loc_loss = F.smooth_l1_loss(xloc[pos_idx].view(-1, 4), loc[pos_idx].view(-1, 4), 
#                                     size_average=False) 

#         # conf
#         if self.onehot is None or self.onehot.size() != xconf.size():
#             self.onehot = Variable(torch.zeros(xconf.size())).detach()
#             if xconf.is_cuda:
#                 self.onehot = self.onehot.cuda()
#         self.onehot.data.fill_(0)
#         self.onehot.data.scatter_(-1, label.data.clamp(min=0).long().unsqueeze(-1), 1)

#         pos_idx = pos.unsqueeze(-1).expand_as(xconf)
#         pos_conf_loss = F.multilabel_soft_margin_loss(
#             self.scale * xconf[pos_idx].view(-1, xconf.size(-1)), 
#             self.onehot[pos_idx].view(-1, xconf.size(-1)), 
#             size_average=False)
#         neg_idx = neg.unsqueeze(-1).expand_as(xconf)
#         neg_conf_loss = F.multilabel_soft_margin_loss(
#             self.scale * xconf[neg_idx].view(-1, xconf.size(-1)), 
#             self.onehot[neg_idx].view(-1, xconf.size(-1)), 
#             size_average=False)

#         conf_loss = self.alpha * pos_conf_loss + (1 - self.alpha) * neg_conf_loss

#         N = pos.float().sum().clamp(min=1e-3)
#         return loc_loss / N, conf_loss / N
