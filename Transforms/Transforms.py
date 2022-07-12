from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = r'D:\Pytorch_learning\TensorBoard\hymenoptera_data\train\ants_image\6240338_93729615ec.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')
tensor_trans = transforms.ToTensor()  # tensor_trans 理解为一个自己的工具，来自于totensor工具模板
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
