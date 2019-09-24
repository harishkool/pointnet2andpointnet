import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'ops_utils'))
from pointnet2_util import *

class Pointnet2seg_MSG(nn.Module):
    def __init__(self,num_classes, input_channels=3, use_xyz=True):
        super(Pointnet2MSG,self).__init__()
        self.sa_module = nn.ModuleList()
        self.sa_module.append(
            PointnetSA_MSG(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            sample_list=[16, 32, 128],
            mlp_list=[
                [input_channels, 32, 32, 64],
                [input_channels, 64, 64, 128],
                [input_channels, 64, 96, 128],
            ],
            bn=True,
            use_xyz=True)
        )
        input_channels = 64 + 128 + 128
        self.sa_module.append(PointnetSA_MSG(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            sample_list=[16, 32, 128],
            mlp_list=[
                [input_channels, 32, 32, 64],
                [input_channels, 64, 64, 128],
                [input_channels, 64, 96, 128],
            ],
            bn=True,
            use_xyz=True)
        )
        self.sa_module.append(
            PointnetSA(mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz)
        )
        
        self.interpolation()
        self.unit_pointnet()
        self.interpolation()
        self.unit_pointnet()
        features = input[:,3:]
        input_xyz = input[:,:3]
        
        
    def forward(self, input, features):
        pass