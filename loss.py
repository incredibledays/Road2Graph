import torch
import torch.nn as nn


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def __call__(self, pre, gt):
        seg_loss = self.bce_loss(pre['seg'], gt['seg'])
        # print(seg_loss.item())
        return seg_loss


class TopoLoss(nn.Module):
    def __init__(self):
        super(TopoLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def __call__(self, pre, gt):
        seg_pre = torch.sigmoid(pre['seg'])
        seg_gt = gt['seg']
        seg_loss = torch.mean(self.bce_loss(seg_pre, seg_gt)) * 0.1
        ver_pre = torch.sigmoid(pre['ver'])
        ver_gt = gt['ver']
        ver_loss = torch.mean(self.bce_loss(ver_pre, ver_gt)) * 10
        mid_pre = torch.sigmoid(pre['mid'])
        mid_gt = gt['mid']
        soft_mask = torch.clip(mid_gt + 0.01, 0, 1)
        mid_loss = torch.mean(self.bce_loss(mid_pre, mid_gt)) * 10
        dxy_pre = pre['dxy']
        dxy_gt = gt['dxy']
        dxy_loss = torch.mean(self.smooth_l1(dxy_pre, dxy_gt) * soft_mask) * 1000
        # print(seg_loss.item(), ver_loss.item(), mid_loss.item(), dxy_loss.item())
        return seg_loss + ver_loss + mid_loss + dxy_loss
