import torch
from torch import nn

class TP(nn.Module):
    def __init__(self, weight_loss_TP, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_TP = weight_loss_TP
        self.stopThr = stopThr
        self.epsilon = 1e-6
        self.transp = None


    def forward(self, M, group):
        if self.weight_loss_TP <= 1e-6:
            return 0.
        else:
            device = M.device
            # Sinkhorn's algorithm
            a = (group.sum(axis=1)).unsqueeze(1).to(device)
            b = (group.sum(axis=0)).unsqueeze(1).to(device)

            u = (torch.ones_like(a) / a.size()[0]).to(device) 

            K = torch.exp(-M * self.sinkhorn_alpha).clamp(min=1e-6)
            err = 1
            cpt = 0
            while err > self.stopThr and cpt < self.OT_max_iter:
                v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
                u = torch.div(a, torch.matmul(K, v) + self.epsilon)
                cpt += 1
                if cpt % 50 == 1:
                    bb = torch.mul(v, torch.matmul(K.t(), u))
                    err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

            transp = u * (K * v.T)
            transp = transp.clamp(min=1e-4)

            group = group.clamp(min=1e-6)
            loss_TP = (transp * (transp.log() - group.log() - 1) + group).sum()

            loss_TP *= self.weight_loss_TP

            return loss_TP