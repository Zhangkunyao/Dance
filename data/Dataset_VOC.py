# coding = utf-8
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from data.voc import VocClassification
import torch




# A 使用ＶＯＣ数据集　Ｂ使用任意ＧＴ
class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = opt.voc_dataroot
        self.dir_sal = os.path.join(opt.dataroot, 'sal')

        self.dataset = VocClassification(self.dir, 'VOC2012', opt.phase)
        # B是GT
        self.sal_paths = make_dataset(self.dir_sal)
        self.sal_paths = sorted(self.sal_paths)
        self.sal_size = len(self.sal_paths)
        self.size = self.dataset.__len__()
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index_sal = random.randint(0, self.sal_size - 1)
        sal_path = self.sal_paths[index_sal]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img,label_A = self.dataset.__getitem__(index)
        sal_img = Image.open(sal_path).convert('RGB')

        A = self.transform(A_img)
        sal_img = self.transform(sal_img)
        tmp = sal_img[0, ...] * 0.299 + sal_img[1, ...] * 0.587 + sal_img[2, ...] * 0.114
        sal_img = tmp.unsqueeze(0)
        return {'img': A, 'sal': sal_img,'label': label_A}

    def __len__(self):
        return max(self.size, self.sal_size)

    def name(self):
        return 'UnalignedDataset'

class PathDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'ECSSD')

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')

        A = self.transform(A_img)
        return {'img': A,'img_path': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'

# A 使用ＶＯＣ数据集　Ｂ使用任意ＧＴ
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
class ImagenetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = opt.imagenet_dataroot
        self.dir_sal = os.path.join(opt.dataroot, 'sal')

        self.dataset = datasets.ImageFolder(
        opt.imagenet_dataroot)
        self.name_idx = self.dataset.class_to_idx
        self.name_idx = {int(value): key for (key, value)
                   in self.name_idx.items()}
        # B是GT
        self.sal_paths = make_dataset(self.dir_sal)
        self.sal_paths = sorted(self.sal_paths)
        self.sal_size = len(self.sal_paths)
        self.size = self.dataset.__len__()
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index_sal = random.randint(0, self.sal_size - 1)
        sal_path = self.sal_paths[index_sal]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        # index = random.randint(0, self.size - 1)
        A_img,index_label = self.dataset.__getitem__(index)
        tmp = np.zeros((22), dtype=np.float32)
        tmp[index_label] = 1
        label_A = tmp
        label_A = torch.FloatTensor(label_A)
        # fake_label
        tmp = np.zeros((22), dtype=np.float32)
        if index_label == 20:
            index_label = 0
        else:
            index_label = index_label+1
        tmp[index_label] = 1
        label_fake = tmp
        label_fake = torch.FloatTensor(label_fake)

        sal_img = Image.open(sal_path).convert('RGB')

        A = self.transform(A_img)
        sal_img = self.transform(sal_img)
        tmp = sal_img[0, ...]
        sal_img = tmp.unsqueeze(0)
        return {'img': A, 'sal': sal_img,'label': label_A,'label_fake':label_fake}

    def __len__(self):
        return max(self.size, self.sal_size)

    def name(self):
        return 'ImagenetDataset'

    def class_idx(self):
        return self.name_idx