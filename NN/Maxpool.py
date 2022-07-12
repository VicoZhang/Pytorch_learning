import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets.CIFAR10(
    root=r'D:\Pytorch_learning\TorchVision\dataset',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

data_loader = DataLoader(
    dataset=data_set,
    batch_size=64,
    shuffle=False,
    drop_last=True
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0
        )
        self.maxpol1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            ceil_mode=True
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpol1(x)
        return x


writer = SummaryWriter('logs')
net = Net()
step = 0
for data in data_loader:
    img, target = data
    input = img
    writer.add_images('input', input, step)
    output = net(img)
    output = torch.reshape(output, [-1, 3, 10, 10])
    writer.add_images('output', output, step)
    step += 1

writer.close()
