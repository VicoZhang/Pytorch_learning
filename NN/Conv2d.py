import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root=r'D:\Pytorch_learning\TorchVision\dataset',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
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

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()
writer = SummaryWriter('logs')
step = 0

for data in dataloader:
    imgs, target = data
    input = imgs
    writer.add_images('input', input, step)
    output = net(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)
    step += 1

writer.close()
