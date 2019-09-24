import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transforms import input_transform_net, feature_transform_net
from torch.autograd import Variable
from pcn_utils import vfe


class encoder(nn.Module):
    def __init__(self, npts, gt, alpha):
        super(encoder,self).__init__()
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.mlp1 = torch.nn.Conv1d(3, 128, 1, 1, 0)
        self.mlp2 = torch.nn.Conv1d(128, 256, 1, 1, 0)
        self.mlp3 = torch.nn.Conv1d(256, 512, 1, 1, 0)
        self.mlp4 = torch.nn.Conv1d(512, 1024, 1, 1, 0)

        self.mlp5 = torch.nn.Conv1d(256, 512, 1, 1, 0)
        self.mlp6 = torch.nn.Conv1d(512, 1024, 1, 1, 0)

        self.mlp7 = torch.nn.Conv1d(2048,1024, 1, 1, 0)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(1024)

        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)        


        self.features = self.create_encoder(inputs, npts)
        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        # self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']



    def forward(self, x):
        inputs = x
        b, n, c = x.shape
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        x_m = torch.max(x, 2, keepdim=True)[0]
        x_repeat = x_m.repeat(1, n).view(b, n, -1)
        x = torch.concat([x,x_repeat])
        x = F.relu(self.bn3(self.mlp3(x)))
        x = F.relu(self.bn4(self.mlp4(x)))
        vox = vfe(inputs)
        vox_x_repeat = x_m.repeat(1, vox.shape[1]).view(b, vox.shape[1],-1)
        vox_feat = torch.concat([vox,vox_x_repeat])
        vox_feat = F.relu(self.bn5(self.mlp5(vox_feat)))
        vox_feat = F.relu(self.bn6(self.mlp6(vox_feat)))
        x = torch.concat([vox_feat,x])
        x = F.relu(self.bn7(self.mlp7(x)))


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        




    def forward(self, x_code):
        pass

