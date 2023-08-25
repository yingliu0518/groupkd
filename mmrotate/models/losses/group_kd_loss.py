import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import build_loss
from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def im_loss(x, soft_target):
    # print(x.shape, soft_target.shape)
    # print(F.mse_loss(x, soft_target))
    return F.mse_loss(x, soft_target)

@ROTATED_LOSSES.register_module()
class IMLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                x,
                soft_target,
                group_weight,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = 0.0
        if len(group_weight) == 1:
            for i in range(len(soft_target)):
                loss += self.loss_weight * im_loss(x, soft_target[i], reduction=reduction)
        else:
            for i in range(len(soft_target)):
                loss += self.loss_weight * im_loss(x, soft_target[i], reduction=reduction) * group_weight[0][i]

        # loss_im = self.loss_weight * im_loss(
        #     x, soft_target, reduction=reduction)

        return loss

