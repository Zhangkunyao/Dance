#coding=utf-8
import numpy as np
from Inception_cls import ClassDiscriminator
from voc import VocClassification
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import os
#
# torch.cuda.set_device(3)
#
# def image_show(im,name):
#     im = im.cpu()
#     im = im / 2 + 0.5     # unnormalize
#     npimg = im.numpy()
#     plt.figure(name)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#     # plt.figure(name)
#     # plt.imshow(im_orage)
#     # plt.show()
#     # im_orage.show()
#
# Model = 'val'
# path_basic = "/media/kun/Dataset/PASCAL-VOC/PASCAL-VOC2012/VOCdevkit/"
#
# label_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#
# transform = transforms.Compose(
#     [transforms.Resize(size = (299,299), interpolation=Image.BILINEAR),
#         transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
# dataset = VocClassification(path_basic,'VOC2012',Model,transform = transform)
# nThreads = 4
#
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     shuffle=False,
#     batch_size=16,
#     num_workers=int(nThreads))
#
# clas_model = ClassDiscriminator(3,20,'./Model/image_encoder200.pth').cuda()
# clas_model.load_state_dict(torch.load('./Model/image_encoder200.pth'))
# with torch.no_grad():
#     running_loss =0
#     loss_func = nn.BCELoss()
#     for step, data in enumerate(dataloader, 0):
#         image, label = data
#         image = image.cuda()
#         label = label.cuda()
#         out_label = clas_model(image)
#         loss = loss_func(out_label, label)
#         running_loss += abs(loss.data.cpu())
#
#         # print statistics
#
#         if step % 100 == 0:  # print every 2000 mini-batches
#             tmp_label = out_label.cpu().detach().numpy()
#             true_label = label.cpu().numpy()
#             index = np.argwhere(tmp_label[0] > 0.5)
#             name = ''
#             for i in index:
#                 name += label_name[i[0]] + ' '
#             image_show(image[0],name)
#             print(name)
#             print('[%5d] loss: %f' %
#                   (step + 1, running_loss / 20))
#             print(tmp_label[0])
#             print(true_label[0])
#             running_loss = 0.0
'''
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # 输入的是sal_map一维
            self.netD_A = networks.define_D(opt.output_nc_a, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            # 输入的是梯度图一维
            self.netD_B = networks.define_D(opt.output_nc_a, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            # 输入的是四维　原始图像和sal_map
            self.netD_Mch = networks.define_D(opt.input_nc + opt.output_nc_a, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            # 输入三维　sal_map和原始图像的乘积
            self.netD_Cls = networks.define_C(opt.input_nc,opt.class_num, self.gpu_ids)


101
'''
import random
random.randint(0,10)
loss = nn.L1Loss()
input = torch.Tensor([0,1,0])
print(input)
target = torch.Tensor([1,0,0])
print(target)
output = loss(input, target)
print(output)

# import numpy as np
# import itertools
#
# def combinations(x):
#     num = np.sum(x==1)
#     data_index = range(len(x))
#     out = []
#     print(len(list(itertools.combinations(data_index,num))))
#     print(len(set(list(itertools.combinations(data_index, num)))))
#         # tmp = np.zeros(len(x))
#         # for j in i:
#         #     tmp[j]=1
#         # out.append(tmp)
#         # print(i)
#     return None
#
#
# a=np.concatenate([np.zeros(17),np.ones(3)])
# hehe = combinations(a)
# # for i in itertools.combinations(a,len(a)):
# #     print(i)
