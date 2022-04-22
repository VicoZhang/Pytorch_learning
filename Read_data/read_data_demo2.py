from torch.utils.data import Dataset
from PIL import Image
import os


class ReadData:

    def __init__(self, r_d, i_d, l_d):
        self.root_dir = r_d
        self.image_dir = i_d
        self.label_dir = l_d
        self.image_list = os.listdir(os.path.join(self.root_dir, self.image_dir))
        return

    def __getitem__(self, idx):
        name = self.image_list[idx]
        img_path = os.path.join(root_dir, image_dir, "{}.jpg".format(name[:-4]))
        label_path = os.path.join(root_dir, label_dir, "{}.txt".format(name[:-4]))  # .jpg为4个
        img = Image.open(img_path)
        img.show()
        with open(label_path, 'r') as f:
            label = f.read()
        return name, label


root_dir = 'Read_data/hymenoptera_data/train'
image_dir = 'ants_image'
label_dir = 'ants_label'
ants_dataset = ReadData(root_dir, image_dir, label_dir)
