import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'ops_utils'))
import pointnet2_ops as ops
import util
import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        

class SharedMLP(nn.Module):
    def __init__(self,mlp_spec, bn):
        # pdb.set_trace()
        super(SharedMLP,self).__init__()
        self.convlayers = []
        self.bnlayers =[]
        for i in range(len(mlp_spec)-1):
            mlp_layer = nn.Conv2d(in_channels=mlp_spec[i], out_channels=mlp_spec[i+1], kernel_size=(1,1),
                        stride=(1,1),padding=0,bias=True)
            self.convlayers.append(mlp_layer)
            if bn:
                self.bnlayers.append(nn.BatchNorm2d(mlp_spec[i+1]))
        self.conv1= nn.Conv2d(3,3,(1,1))

    def forward(self, new_features):
        for i in range(len(self.convlayers)-1):
            new_features = F.relu(self.bnlayers[i](self.convlayers[i](new_features)))
        return new_features


class PointnetSA_MSG(nn.Module):
    def __init__(self, npoint, radius_list, sample_list, mlp_list, bn=True, use_xyz=True):
        super(PointnetSA_MSG,self).__init__()
        self.radius_list = radius_list
        self.sample_list = sample_list
        self.npoint = npoint
        self.mlps = nn.ModuleList()
        self.groupers = nn.ModuleList()
        for idx,val in enumerate(self.radius_list):
            self.groupers.append(ops.QueryAndGroup(val, self.sample_list[idx]))
            mlp_spec = mlp_list[idx]
            if use_xyz:
                mlp_spec[0]+=3
            self.mlps.append(SharedMLP(mlp_spec,bn))

    def forward(self, input, features):
        msg_features_list = []
        input_flipped = input.transpose(1,2).contiguous()
        new_xyz = ops.gather_points(input_flipped, ops.farthest_point_sampling(input, self.npoint)).transpose(1,2).contiguous()
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](input_flipped, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            msg_features_list.append(new_features)
        return new_xyz, torch.cat(msg_features_list,1)


class PointnetSA(nn.Module):
    def __init__(self, mlp, bn=True):
        super(PointnetSA,self).__init__()
        self.mlps = nn.ModuleList()
        for i in range(len(mlp)-1):
            mlp_spec = mlp[i]
            self.mlps.append(SharedMLP(mlp_spec,bn))

    def forward(self, features):
        for i in range((self.mlps)-1):
            features = self.mlps[i](features)
        return features
        



if __name__ == "__main__":
    # pdb.set_trace()
    # train_data, current_labels = util.loadDataFile('../data/modelnet40/ply_data_train1.h5')
    # train_data = util.to_var(torch.from_numpy(train_data))
    # current_labels = util.to_var(torch.from_numpy(current_labels))
    # num_batches = train_data.shape[0] // 32
    # current_train = train_data[0].cuda()
    # npoints = 512
    # current_train_flipped = current_train.reshape(1,current_train.shape[1], current_labels.shape[0])
    # current_train = current_train.unsqueeze(0)
    # sampling_output = ops.farthest_point_sampling(current_train,npoints)
    # gatherpoints = ops.gather_points(current_train_flipped, sampling_output)
    # nsample = 16
    # radius =0.1
    # idx = ops.ball_query(radius, nsample, current_train, gatherpoints)
    # grouped_xyz = ops.grouping_operation(current_train_flipped, idx)
    # print(grouped_xyz.shape)
    x = torch.randn((1,32,3,32))
    mlp_list = [32,64,128]
    model = SharedMLP(mlp_list,True)
    print(F.relu(model.bnlayers[0](model.convlayers[0](x))))