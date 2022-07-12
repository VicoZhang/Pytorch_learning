import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 方法一
torch.save(vgg16, 'vgg16.pth')


# 方法二
torch.save(vgg16.state_dict(), "vgg16-2.pth")
