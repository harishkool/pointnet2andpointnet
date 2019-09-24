import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transforms import input_transform_net, feature_transform_net
from torch.autograd import Variable
from transforms_alt import input_transform_net_alt, feature_transform_net_alt

def mlp(in_channels, out_channels, filter_size, padding=1, stride=1,batch_norm=False,init_weights=False):
    layers = []
    mlp_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
    if init_weights:
        mlp_layer.weight.data = (torch.randn(out_channels, in_channels, filter_size[0], filter_size[1]) * 0.001)
    layers.append(mlp_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def mlp_conv1d(in_channels, out_channels, filter_size, padding=1, stride=1,batch_norm=False,init_weights=False):
    layers = []
    mlp_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
    if init_weights:
        mlp_layer.weight.data = (torch.randn(out_channels, in_channels, filter_size[0], filter_size[1]) * 0.001)
    layers.append(mlp_layer)

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)

def maxpool(filter_size, stride=2, padding=0):
    return nn.MaxPool2d(filter_size, stride, padding)

class PointNetcls(nn.Module):
    def __init__(self, num_classes):
        super(PointNetcls,self).__init__()
        self.mlp1 = mlp(1, 64, (1,3), 0, 1)
        self.mlp2 = mlp(64, 64, (1,1), 0, 1)
        self.mlp3 = mlp(64, 128, (1,1), 0, 1)
        self.mlp4 = mlp(128, 1024, (1,1), 0, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.last = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        Input_transform = input_transform_net()
        transform = Input_transform(x)
        # pdb.set_trace()
        x = torch.matmul(x, transform)
        bn, inpt, pts = x.shape[0], x.shape[1], x.shape[2]
        x = x[:,:,:,None]
        x = x.view(bn, 1, inpt, pts).type(torch.cuda.FloatTensor)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        Featuretransform = feature_transform_net()
        feature_transform = Featuretransform(x)
        bn, pts, inpt = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(bn, inpt, pts)
        x = torch.matmul(x, feature_transform)
        x = x.view(bn, pts, inpt, 1)       
        x = F.relu(self.mlp3(x))
        x = F.relu(self.mlp4(x))
        maxpool2d = maxpool([inpt, 1])
        x = maxpool2d(x)
        # x = torch.max(x,2)[0]
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])
        x = F.relu(self.bn1(self.dp1(self.fc1(x))))
        x = F.relu(self.bn2(self.dp2(self.fc2(x))))
        x = F.relu(self.last(x))
        # pdb.set_trace()
        return x.view(x.shape[0], x.shape[1])


class PointNetcls_conv1d(nn.Module):
    def __init__(self, num_classes):
        super(PointNetcls_conv1d,self).__init__()
        # self.mlp1 = mlp_conv1d(3, 64, 1, 0, 1)
        # self.mlp2 = mlp_conv1d(64, 64, 1, 0, 1)
        # self.mlp3 = mlp_conv1d(64, 128, 1, 0, 1)
        # self.mlp4 = mlp_conv1d(128, 1024, 1, 0, 1)
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.last = nn.Linear(256, num_classes)

        self.mlp1 = torch.nn.Conv1d(3, 64, 1, 1, 0)
        self.mlp2 = torch.nn.Conv1d(64, 64, 1, 1, 0)
        self.mlp3 = torch.nn.Conv1d(64, 128, 1, 1, 0)
        self.mlp4 = torch.nn.Conv1d(128, 1024, 1, 1, 0)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.last = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6  = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # x = x.type(torch.cuda.FloatTensor)
        Input_transform = input_transform_net_alt().cuda()
        transform = Input_transform(x)
        # pdb.set_trace()
        # if x.is_cuda:
        #    transform=transform.cuda()
        x = torch.bmm(x, transform)
        bn, inpt, pts = x.shape[0], x.shape[1], x.shape[2]
        # x = x[:,:,:,None]
        x = x.view(bn, pts, inpt)
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        # pdb.set_trace()
        Featuretransform = feature_transform_net_alt().cuda()
        feature_transform = Featuretransform(x)
        bn, pts, inpt = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(bn, inpt, pts)
        # if x.is_cuda:
        #     feature_transform =feature_transform.cuda()
        x = torch.bmm(x, feature_transform)
        x = x.view(bn, pts, inpt)       
        x = F.relu(self.bn3(self.mlp3(x)))
        x = F.relu(self.bn4(self.mlp4(x)))
        # maxpool2d = maxpool([inpt, 1])
        # x = maxpool2d(x)
        x = torch.max(x,2,keepdim=True)[0]
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])
        x = F.relu(self.bn5(self.dp1(self.fc1(x))))
        x = F.relu(self.bn6(self.dp2(self.fc2(x))))
        x = self.last(x)
        # pdb.set_trace()
        return x.view(x.shape[0], x.shape[1])