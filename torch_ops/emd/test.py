import torch
import torch.nn as nn
from torch.autograd import Variable
import dist_emd as emd


p1 = torch.rand(10,1000,6)
p2 = torch.rand(10,1500,6)
points1 = Variable(p1,requires_grad = True).cuda()
points2 = Variable(p2, requires_grad = True).cuda()
cost = emd.emdDist(points1, points2)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(points1.grad, points2.grad)


# points1 = Variable(p1, requires_grad = True)
# points2 = Variable(p2, requires_grad = True)
# cost = emd.emdDist(points1, points2)
# print(cost)
# loss = torch.sum(cost)
# print(loss)
# loss.backward()
# print(points1.grad, points2.grad)
