#coding=utf-8
import numpy as np
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
import torchvision
import torchvision.transforms as transforms

class ClassDiscriminator(nn.Module):
    def __init__(self, input_nc, Cls_Num):
        super(ClassDiscriminator, self).__init__()
        self.input_dim = input_nc
        self.out_dim = Cls_Num
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))

        for param in model.parameters():
            param.requires_grad = True
        print('Load pretrained model from ', url)
        # print(model)
        self.mode = self.define_module(model)
        self.load_state_dict(torch.load('./Model/image_encoder200.pth'))
        # nn.init.normal_(self.parameters(), mean=0, std=1)
    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        # for i in self.Mixed_7c.parameters():
        #     i.requires_grad = True
        # 把最后的ＦＣ层最后的输出类别换成ＶＯＣ的类别
        self.Liner = nn.Linear(2048, self.out_dim)
        for i in self.Liner.parameters():
            i.requires_grad = True
        self.Liner.stddev = 0.01


    def forward(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # image region features
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        # 1000 (num_classes)
        label_out = F.sigmoid(self.Liner(x))
        return label_out
def image_show(im,name):
    im = im.cpu()
    im = im / 2 + 0.5     # unnormalize
    npimg = im.numpy()
    plt.figure(name)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # plt.figure(name)
    # plt.imshow(im_orage)
    # plt.show()
    # im_orage.show()

def Get_List(path):
    files = os.listdir(path);
    dirList = []
    fileList = []
    for f in files:
        if (os.path.isdir(path + '/' + f)):

            if (f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            fileList.append(f)
    return [dirList, fileList]

clas_model = ClassDiscriminator(3,20,).cuda()
# clas_model.load_state_dict(torch.load('./Model/image_encoder200.pth'))


Model = 'train'
path_basic = "/media/kun/Dataset/PASCAL-VOC/PASCAL-VOC2012/VOCdevkit/"

label_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

transform = transforms.Compose(
    [transforms.Resize(size = (299,299), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

dataset = VocClassification(path_basic,'VOC2012',Model,transform = transform)
nThreads = 4

dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=False,
    batch_size=16,
    num_workers=int(nThreads))

torch.cuda.set_device(3)

pra = []
lr = 0.001
for i in clas_model.parameters():
    if i.requires_grad == True:
        pra.append(i)

if Model == 'train':
    loss_func = nn.BCELoss()
    running_loss = 0
    for epoch in range(201):
        optimizer = optim.Adam(pra, lr=lr, betas=(0.5, 0.999))
        for step, data in enumerate(dataloader, 0):
            clas_model.zero_grad()
            image, label= data
            image = image.cuda()
            label = label.cuda()
            out_label = clas_model(image)
            loss = loss_func(out_label, label)
            running_loss += abs(loss.data.cpu())
            loss.backward()
            optimizer.step()
            # print statistics

            if step % 103 == 0:    # print every 2000 mini-batches
                tmp_label = out_label.cpu().detach().numpy()
                true_label = label.cpu().numpy()
                index = np.argwhere(tmp_label[0]>0.5)
                name = ''
                for i in index:
                    name += label_name[i[0]]+' '

                # image_show(image[0],name)
                print(name)
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, step + 1, running_loss / 20))
                print(tmp_label[0])
                print(true_label[0])
                running_loss = 0.0
        lr = lr*0.98
        if epoch%50==0:
            torch.save(clas_model.state_dict(),
                       './Model/image_encoder_last.pth' % (epoch))

if Model == 'val' or 'test':
    with torch.no_grad():
        running_loss =0
        model_save_path = './Model/'
        _,flie_all = Get_List(model_save_path)
        # flie_all.sort()
        model_path = model_save_path + 'image_encoder200.pth'
        clas_model.load_state_dict(torch.load(model_path))
        loss_func = nn.BCELoss()
        for step, data in enumerate(dataloader, 0):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            out_label = clas_model(image)
            loss = loss_func(out_label, label)
            running_loss += abs(loss.data.cpu())

            # print statistics

            if step % 100 == 0:  # print every 2000 mini-batches
                tmp_label = out_label.cpu().detach().numpy()
                true_label = label.cpu().numpy()
                index = np.argwhere(tmp_label[0] > 0.5)
                name = ''
                for i in index:
                    name += label_name[i[0]] + ' '

                image_show(image[0],name)
                print(name)
                print('[%5d] loss: %f' %
                      (step + 1, running_loss / 20))
                print(tmp_label[0])
                print(true_label[0])
                running_loss = 0.0


