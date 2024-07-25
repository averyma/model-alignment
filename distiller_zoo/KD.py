from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class SymmetricKL(nn.Module):
    '''This is my own implementation of distillation using kl'''
    def __init__(self):
        super(SymmetricKL, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p_s, p_w):
        yp_s = F.log_softmax(p_s, dim=1)
        yp_w = F.log_softmax(p_w, dim=1)
        loss = self.kl_loss(yp_s, yp_w) + self.kl_loss(yp_w, yp_s)
        return loss
