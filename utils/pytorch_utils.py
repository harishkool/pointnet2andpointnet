import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, mlp_list):
        super(MLP, self).__init__()
