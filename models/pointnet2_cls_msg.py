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

class Pointnet2cls_MSG(nn.Module):
    def __init__(self,num_classes, input_channels=3, use_xyz=True):
        super(Pointnet2cls_MSG,self).__init__()
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
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            sample_list=[16, 32, 128],
            mlp_list=[
                [input_channels, 32, 32, 128],
                [input_channels, 64, 64, 256],
                [input_channels, 64, 96, 256],
            ],
            bn=True,
            use_xyz=True)
        )
        self.sa_module.append(
            PointnetSA(mlp=[128 + 256 + 256, 256, 512, 1024])
        )

        self.fc_layer1 = nn.Linear(1024, 512)
        self.drop1 = nn.Dropout(0.2)
        self.fc_layer2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.4)
        self.fc_layer3 = nn.Linear(256, num_classes)
         
    def get_features(self, pointcloud):
        features = input[:,3:].contiguous().transpose(1,2) if pointcloud.size(-1)>3 else None
        input_xyz = input[:,0:3]
        return input_xyz, features

    def forward(self, pointcloud):
        xyz, features = self.get_features(pointcloud)
        for sa in self.sa_module:
            xyz, features = sa(xyz, features)
        features = self.drop1((self.fc_layer1(features)))
        features = self.drop2((self.fc_layer2(features)))
        logits = self.fc_layer3(features)
        return logits