###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import os
import os.path
import random
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images)


def default_loader(path):
    img = Image.open(path).convert('RGB')
    # img = img.rotate(270)
    return img


class ImageFolder(BaseDataset):

    def __init__(self,opt,root):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)

        self.imgs = make_dataset(os.path.join(root, 'img'))
        self.pose = make_dataset(os.path.join(root, 'pose'))

        self.transform = self.transform
        self.loader = default_loader

    def __getitem__(self, index):
        shuffe_index = random.randint(0, len(self.imgs) - 1)
        while shuffe_index==index:
            shuffe_index = random.randint(0, len(self.imgs) - 1)
        imgs_path = self.imgs[index]
        pose_path = self.pose[index]
        rand_pose_path = self.pose[shuffe_index]
        rand_img_path = self.imgs[shuffe_index]
        img = self.loader(imgs_path)
        pose = self.loader(pose_path)
        rand_pose = self.loader(rand_pose_path)
        rand_img = self.loader(rand_img_path)
        if self.transform is not None:
            img = self.transform(img)
            pose = self.transform(pose)
            rand_pose = self.transform(rand_pose)
            rand_img = self.transform(rand_img)
        return {'img': img, 'pose': pose,'rand_pose':rand_pose,'rand_img':rand_img}

    def __len__(self):
        return len(self.imgs)
