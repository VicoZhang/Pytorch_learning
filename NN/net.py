import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    @staticmethod
    def forward(input):
        output = input + 1
        return output


net = Net()
x = torch.tensor(1.0)
out = net(x)
print(out)