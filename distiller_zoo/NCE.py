import torch
import torch.nn.functional as F
from torch import nn

EPISILON=1e-10

"CCL original loss"

class NCELoss(torch.nn.Module):

  def __init__(self, temperature=0.1):
    super(NCELoss, self).__init__()
    self.temperature = temperature
    self.softmax = nn.Softmax(dim=1)

  def forward(self, f1, f2, targets):
     
    ### cuda implementation
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)
    #print('f1.shape', f1.shape)
    #print('f2.shape', f2.shape)

    ## set distances of the same label to zeros
    mask = targets.unsqueeze(1) - targets
    #print(targets.unsqueeze(1))
    #print('mask', mask)
    self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
    #print('self_mask', self_mask)
    self_mask = 0 + self_mask
    dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

    ## convert l2 distance to cos distance
    cos = 1 - 0.5 * dist
    
    ## convert cos distance to exponential space
    pred_softmax = self.softmax(cos / self.temperature) ### convert to multi-class prediction scores
    #pred_softmax size: (n,n)

    log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
    log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
    log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(1).float()
    loss = log_softmax
    #loss size: (batch_size,)

    return loss.mean()
