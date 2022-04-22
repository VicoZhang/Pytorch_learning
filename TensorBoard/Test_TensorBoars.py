from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = 'hymenoptera_data/train/ants_image/9715481_b3cb4114ff.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
writer.add_image('test1', img_array, 2, dataformats='HWC')  # dataformats='HWC'修改颜色通道的顺序

for i in range(100):
    writer.add_scalar('y = 2x', i, i)

writer.close()
