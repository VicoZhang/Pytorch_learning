import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10(r'D:\Pytorch_learning\TorchVision\dataset',
                                          train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

print(vgg16_true)

# 修改方法一
vgg16_true.classifier.add_module('7', nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)

# 修改方法二
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_false)