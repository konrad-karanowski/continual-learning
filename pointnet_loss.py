import torch
from torch.nn import functional as F


def pointnetloss(y_hat, y, m3x3, m64x64, alpha = 0.0001):
    bs=y_hat.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if y_hat.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return F.cross_entropy(input=y_hat, target=y.long(), reduction='mean') + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)
