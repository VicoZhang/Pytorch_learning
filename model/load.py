# 方法一
import torch

# vgg16 = torch.load('vgg16.pth')
# print(vgg16)

# 方法二
import torchvision

vgg16_2 = torchvision.models.vgg16(pretrained=False)
vgg16_2.load_state_dict(torch.load('vgg16-2.pth'))

print(vgg16_2)