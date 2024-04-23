import torch
from torch import nn
from models.triplet_loss import HardTripletLoss
import torch.nn.functional as F
import numpy as np

class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


def choose_align(type_id, n, device, b):
    if type_id == 0:
        return torch.arange(n, device=device).repeat(b)            # 0 - n
    elif type_id == 1:
        return (torch.arange(-n//2,n//2, 1, device=device)/(n//2)).repeat(b) # {[-1,1),[-1,1),[-1,1),[-1,1)}
    else:
        n = n * b
        return (torch.arange(-n//2,n//2, 1, device=device)/(n//2)) # -1, 1
# 
class LossFun(nn.Module):
    def __init__(self, alpha, margin, etf_head = False, loss_align = 0):
        super(LossFun, self).__init__()
        if etf_head:
            self.dr_loss = nn.L1Loss() # DRLoss()
        else:
            self.dr_loss = nn.MSELoss()
        self.loss_align = loss_align
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.beta = 10

    def forward(self, pred, label, feat):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            t_loss = 0
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = choose_align( self.loss_align, n, device, b)
            t_loss = self.triplet_loss(flat_feat, la) # t_loss = 0
           
        else:
            self.alpha = 0
            t_loss = 0

  
        if type(pred) is list:
          
            dr_loss = 0
            
            for i in range(len(pred)):
                dr_loss += self.dr_loss(pred[i], label[i])
               
        elif type(pred) is dict:
            dr_loss = 0
            dr_loss += self.mse_loss(pred['int'], label[0])
            offset = 1
            if 'int_revert' in pred:
                dr_loss += self.mse_loss(pred['int_revert'], label[1])
                offset +=1
            for i in range(len(pred['dec'])):
                dr_loss += self.dr_loss(pred['dec'][i], label[i+offset])
            
        else:
            dr_loss = self.dr_loss(pred, label)
     
        return dr_loss + self.alpha * t_loss, dr_loss, t_loss
      
