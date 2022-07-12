from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img_PLI = Image.open(r'D:\Pytorch_learning\Transforms\1618673847750.jpeg')
# print(img)


# ToTensor
trans_ToTensor = transforms.ToTensor()
img_tensor = trans_ToTensor(img_PLI)
writer.add_image('ToTensor', img_tensor, 1)

# Normalize
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image('norm', img_norm, 4)


# Resize_1
trans_resize = transforms.Resize([512, 512])
img_resize_1 = trans_resize(img_PLI)
img_resize_1 = trans_ToTensor(img_resize_1)
writer.add_image('resize', img_resize_1, 0)

# Resize_2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_ToTensor])
img_resize_2 = trans_compose(img_PLI)
writer.add_image('resize', img_resize_2, 1)


# RandomCrop
trans_randomcrop = transforms.RandomCrop([100, 512])
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_ToTensor])
for i in range(10):
    img_random = trans_compose_2(img_PLI)
    writer.add_image('RandomCrop_2', img_random, i)

writer.close()

