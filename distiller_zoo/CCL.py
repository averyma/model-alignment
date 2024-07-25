import torch
import torch.nn.functional as F
from torch import nn

EPISILON=1e-10

class JSDLoss(torch.nn.Module):

  def __init__(self, T):
    super(JSDLoss, self).__init__()
    self.T = T

  def forward(self, y_s, y_t):
    p_s = F.log_softmax(y_s/self.T, dim=1)
    p_t = F.softmax(y_t/self.T, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]

    p_t = F.log_softmax(y_t/self.T, dim=1)
    p_s = F.softmax(y_s/self.T, dim=1)
    loss_sys = F.kl_div(p_t, p_s, size_average=False) * (self.T**2) / y_t.shape[0]

    return 0.5 * (loss + loss_sys)


class nce(torch.nn.Module):

  def __init__(self, temperature=1):
    super(nce, self).__init__()
    self.temperature = temperature
    self.softmax = nn.Softmax(dim=1)

  def where(self, cond, x_1, x_2):
    cond = cond.type(torch.float32)
    return (cond * x_1) + ((1 - cond) * x_2)

  def forward(self, f1, f2, targets):
    ### cuda implementation
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    ## set distances of the same label to zeros
    mask = targets.unsqueeze(1) - targets
    self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
    dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

    ## convert l2 distance to cos distance
    cos = 1 - 0.5 * dist

    ## convert cos distance to exponential space
    pred_softmax = self.softmax(cos / self.temperature) ### convert to multi-class prediction scores

    log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
    log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
    log_softmax = log_pos_softmax.sum(1) / (1 - self_mask.float()).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(1).float()
    loss = log_softmax

    return loss.mean()



class CCL(torch.nn.Module):

  def __init__(self, temperature=1):
    super(CCL, self).__init__()
    self.nceLoss = nce(temperature = 0.1)
    self.jsdLoss = JSDLoss(T = 1)

  def forward(self, fs, ft, target, logit_s, logit_t):
    
    nceloss = self.nceLoss(fs, ft, target)
    jsdloss = self.jsdLoss(logit_s, logit_t)
    #print('nce',nceloss)
    #print('jsd', jsdloss)

    loss = nceloss + jsdloss
    return loss

    