import math
from torch import nn
from torch.autograd import Function
import torch
import sys
from numbers import Number
from collections import Set, Mapping, deque
import emd


# Earth mover distance module @harishcancan
# GPU tensors only
class emdFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        
        match = emd.approxmatch_forward(xyz1, xyz2)
        cost = emd.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, a=None, b=None):
        xyz1, xyz2, match= ctx.saved_tensors
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        emd.matchcost_backward(xyz1, xyz2, match, gradxyz1, gradxyz2)
        return gradxyz1, gradxyz2

emdDist = emdFunction.apply

