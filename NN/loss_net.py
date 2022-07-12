import torchvision
from torch import nn

from torch.utils.data import DataLoader

data_set = torchvision.datasets.CIFAR10(r'D:\Pytorch_learning\TorchVision\dataset',
                                        train=False, transform=torchvision.transforms.ToTensor(),
                                        download=False)

data_loader = DataLoader(dataset=data_set, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5),
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5,),
                      padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                      padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        y = self.module(x)
        return y


loss = nn.CrossEntropyLoss()
net = Net()

for data in data_loader:
    imgs, target = data
    outputs = net(imgs)
    result_loss = loss(outputs, target)
    result_loss.backward()
    # print(result_loss)
