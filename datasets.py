import glob
import random
import os
import cv2
import numpy as np

# import ISAR_scatter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
# import Opt_PCA


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


####### 读取图像文件 #######
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode + '\*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode + '\*.*')))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")

        if self.unaligned:
            number = random.randint(0, len(self.files_B) - 1)
            image_B = Image.open(self.files_B[number]).convert("RGB")
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset_opt(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.path_img = root
        self.files_img = os.listdir(self.path_img)
        # self.files_img.sort(key=lambda x: int(x.split('.')[0].replace(' ', '')))

    def __getitem__(self, index):
        file_name = self.files_img[index % len(self.files_img)]
        image_A = Image.open(self.path_img + '/' + self.files_img[index % len(self.files_img)]).convert("RGB")
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        item_A = self.transform(image_A)
        return {"A": item_A, "file_name": file_name[:-4]}

    def __len__(self):
        return len(self.files_img)


class ImageDataset_bi_class(Dataset):
    def __init__(self, root, mode="train", transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        txt = open(self.root + '/' + '%s.txt' % mode, 'r')
        self.txt = txt.readlines()
        self.label = []
        for i in range(len(self.txt)):
            self.label.append(self.txt[i][-2])
            self.txt[i] = self.txt[i][:-3]
        # self.files = sorted(glob.glob(self.root + '\*.jpg'))
        self.files = sorted(glob.glob(self.root + '\*.png'))
        # a = 0

    def __getitem__(self, index):
        # c = len(self.files)
        file_name = self.files[index % len(self.files)]
        a = self.txt.index(file_name)
        label = int(self.label[a])
        image_A = Image.open(file_name).convert("RGB")
        item_A = self.transform(image_A)

        return {"A": item_A, "label": label, "file_name": file_name[:-4]}

    def __len__(self):
        return len(self.files)


class ImageDataset_multi_class(Dataset):
    def __init__(self, root, mode="train", transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        self.label = []
        self.files = sorted(os.listdir(self.root))
        a = 0

    def __getitem__(self, index):
        label = int(index % len(self.files))
        img_files = sorted(os.listdir(self.root + '/' + str(label)))
        num = np.random.randint(0, len(img_files))
        file_name = img_files[num]
        image_A = Image.open(self.root + '/' + str(label) + '/' + file_name).convert("RGB")
        item_A = self.transform(image_A)

        return {"A": item_A, "label": label, "file_name": file_name[:-4]}

    def __len__(self):
        return len(self.files)