import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def mlp(in_channels, out_channels, filter_size, stride, padding, bias=False, batch_norm=True, init_weights=False):
    layers = []
    mlp_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, stride=stride, padding=padding, bias=False)
    if init_weights:
        mlp_layer.weight.data = (torch.randn(out_channels, in_channels, filter_size[0], filter_size[1]) * 0.001).type(torch.cuda.FloatTensor).to('cuda')
    mlp_layer.to('cuda')
    layers.append(mlp_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels).cuda())
    return nn.Sequential(*layers)



def maxpool(filter_size, stride=2, padding=0):
    return nn.MaxPool2d(filter_size, stride, padding)


class input_transform_net(nn.Module):
    def __init__(self):
        super(input_transform_net, self).__init__()
        self.mlp1 = mlp(1, 64, (1,3), 1, 0)
        self.mlp2 = mlp(64, 128, (1,1), 1, 0)
        self.mlp3 = mlp(128, 1024, (1,1), 1, 0)
        self.fc1 = nn.Linear(1024, 512).cuda()
        self.fc2 = nn.Linear(512, 256).cuda()

    def forward(self,x,k=3):
        bn, inpt, pts = x.shape[0], x.shape[1], x.shape[2]
        # pdb.set_trace()
        x = x[:,:,:,None]
        x = x.view(bn, 1, inpt, pts)
        x.to('cuda')
        x = F.relu(self.mlp1(x)).cuda()
        x = F.relu(self.mlp2(x)).cuda()
        x = F.relu(self.mlp3(x)).cuda()
        maxpool2d = maxpool([inpt, 1])
        # x = torch.max(x,2)[0].cuda()
        x = maxpool2d(x)
        # print('After maxpool: {}'.format(x.shape))
        x = x.view(bn, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        weights = Variable(torch.zeros((256, 3*k))).cuda()
        transform = torch.matmul(x,weights)
        bias = Variable(torch.zeros(3*k)).type(torch.cuda.FloatTensor)
        bias+= torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1])).type(torch.cuda.FloatTensor)
        transform = torch.add(transform, bias)
        transform = transform.view(bn, k, k)
        return transform


class feature_transform_net(nn.Module):

    def __init__(self):
        super(feature_transform_net, self).__init__()
        self.mlp1 = mlp(64, 64, (1,1) , 1, 0)
        self.mlp2 = mlp(64, 128, (1,1), 1, 0)       
        self.mlp3 = mlp(128, 1024, (1,1), 1, 0)
        self.fc1 = nn.Linear(1024, 512).cuda()
        self.fc2 = nn.Linear(512, 256).cuda()

    def forward(self,x,K=64):
        # pdb.set_trace()
        bn, pts, inpt = x.shape[0], x.shape[1], x.shape[2]   # bn x 1 x 2048 x 64
        x = x[:, :, :, None]
        x = x.view(bn, pts, inpt, 1)
        x.to('cuda')
        x = F.relu(self.mlp1(x)).cuda()
        x = F.relu(self.mlp2(x)).cuda()
        x = F.relu(self.mlp3(x)).cuda()
        # x = torch.max(x,2)[0]
        maxpool2d = maxpool([inpt, 1])
        x = maxpool2d(x)
        # print('After maxpool: {}'.format(x.shape))
        x = x.view(bn, 1024)
        x = self.fc1(x)
        x = self.fc2(x)
        # pdb.set_trace()
        weights = Variable(torch.zeros((256, K*K))).cuda()
        transform = torch.matmul(x,weights)
        transform = transform.view(bn,K,K)
        bias = Variable(torch.eye(K)).cuda()
        transform = torch.add(transform, bias)
        return transform