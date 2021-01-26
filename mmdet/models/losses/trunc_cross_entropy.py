import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module
class TrunCrossEntropyLoss(nn.Module):
    def __init__(self, loss_trunc_thr=0.5, decay=0.1, ignore_index=-1):
        super(TrunCrossEntropyLoss, self).__init__()
        self.loss_trunc_thr = loss_trunc_thr
        self.decay = decay
        self.ignore_index = ignore_index

    def forward(self, input, targets):
        weights = (targets > self.ignore_index).float()
        normalizer = max(1, weights.sum().data.cpu())

        p = F.softmax(input, dim=1)
        p_max = p[:, 1:].max(dim=1)[0]
        inds = (p_max > self.loss_trunc_thr) & (targets == 0)
        weights[inds] = self.decay

        loss = F.cross_entropy(input, targets, reduce=False)
        loss = loss * weights.cuda() / normalizer.cuda()
        return loss.sum()
