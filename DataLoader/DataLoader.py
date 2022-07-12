import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(
    root=r'D:\Pytorch_learning\TorchVision\dataset',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,  # 每次取的数量
    shuffle=False,  # 是否打乱数据
    num_workers=0,  # 单进程
    drop_last=True
)

# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter('dataloader')
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images('epoch:{}'.format(epoch), imgs, step)
        step += 1

writer.close()
