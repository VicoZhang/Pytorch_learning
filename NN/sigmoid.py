from datetime import datetime

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10(r'D:\Pytorch_learning\TorchVision\dataset',
                                        train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)

data_loader = DataLoader(dataset=data_set,
                         batch_size=64,
                         shuffle=True,
                         drop_last=True
                         )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1),
                               padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        y = self.maxpool1(x)
        return y


TIMESTAMP = "logs_{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
logs_dir = TIMESTAMP
net = Net()
writer = SummaryWriter(logs_dir)
step = 0

for data in data_loader:
    img, target = data
    output = net(img)
    writer.add_images(tag='input', img_tensor=img, global_step=step)
    output = torch.reshape(output, [-1, 3, 5, 5])
    writer.add_images(tag='output', img_tensor=output, global_step=step)
    step += 1

writer.close()