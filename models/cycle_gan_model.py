# coding=utf-8
import torch
from torch.autograd import Variable
import itertools
from .base_model import BaseModel
from . import networks
from . import ConvCRF
import numpy as np
import torch
import cv2
import torch.nn.functional as F


def pram_gard(moudel):
    par = []
    for i in moudel.parameters():
        if i.requires_grad == True:
            par.append(i)
    return par

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['D_tf', 'G','Cycle']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            visual_names_A = ['input_img', 'input_pose', 'fake']
        else:
            visual_names_A = ['input_img', 'input_pose', 'fake']

        self.visual_names = visual_names_A

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['D_tf', 'G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # G_Real 生成真实图像
        # 输入 pose 和 img
        self.netG = networks.define_G(3, 3,
                                          64, 'unet_256', opt.norm, not opt.no_dropout, opt.init_type,
                                          gpu_ids=self.gpu_ids)



        if self.isTrain:
            use_sigmoid = True

            self.netD_tf = networks.define_D(3, 64, opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionGANCycle = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, l1use=True)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionClass = torch.nn.BCELoss()
            # initialize optimizers
            # RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))  # alpha=0.9 Adam betas=(opt.beta1, 0.999)

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_tf.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))  # alpha=0.9 Adam betas=(opt.beta1, 0.999)

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))  # 确定优化策略，多少个step更改学习率

        # if not self.isTrain or opt.continue_train:
        self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def set_input(self, input):
        if self.isTrain:
            input_img = input['img']
            input_pose = input['pose']
            if len(self.gpu_ids) > 0:
                input_img = input_img.cuda(self.gpu_ids[0], async=True)
                input_pose = input_pose.cuda(self.gpu_ids[0], async=True)
            self.input_img = input_img
            self.input_pose = input_pose

        else:
            input_img = input['img']
            input_pose = input['pose']
            if len(self.gpu_ids) > 0:
                input_img = input_img.cuda(self.gpu_ids[0], async=True)
                input_pose = input_pose.cuda(self.gpu_ids[0], async=True)
            self.input_img = input_img
            self.input_pose = input_pose

    def get_pose(self):
        return self.input_pose

    def get_img(self):
        return self.input_img

    def set_mode(self, train):
        self.isTrain = train

    def forward(self):
        if self.isTrain:
            self.input_img = Variable(self.input_img)
            self.input_pose = Variable(self.input_pose)
        else:
            self.input_img = Variable(self.input_img)
            self.input_pose = Variable(self.input_pose)

    def test(self):
        self.fake = self.netG(self.input_pose)
        return self.fake

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_D

    def backward_D(self, flag_shuffer=False):

        if flag_shuffer == False:
            input_truth = self.input_img
            input_false = self.fake
            self.loss_D_tf = self.backward_D_basic(self.netD_tf, input_truth,input_false)

            self.loss_all = (self.loss_D_tf)
            self.loss_all.backward()
        else:
            input_truth = self.fake
            input_false = self.input_img
            self.loss_D_tf = self.backward_D_basic(self.netD_tf, input_truth,input_false)

            self.loss_all = (self.loss_D_tf)
            self.loss_all.backward()

    def backward_G(self):
        # 模型前向传播
        self.fake = self.netG(self.input_pose)

        self.loss_Cycle = self.criterionCycle(self.fake, self.input_img)
        # fake 判断
        self.loss_G = self.criterionGAN(self.netD_tf(self.fake), True)

        # loss计算
        self.loss_all = self.loss_G + self.loss_Cycle
        self.loss_all.backward()

    def optimize_parameters(self, flag_shuffer=False, use=False):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


class CRF_GAN(BaseModel):
    def name(self):
        return 'CRF_GAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.features_blobs = []
        self.features_blobs_Noise = []
        # 取消D_sal 直接生成salmap 最后一层转为softmax层
        # 判断真假 分类loss
        self.loss_names = ['Cycle']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            visual_names_A = ['input_img', 'sal']
            visual_names_middle = ['look']
            visual_names_A += ['CRF_Out_look']
        else:
            visual_names_A = ['input_img', 'sal']
            visual_names_middle = ['look']

        self.visual_names = visual_names_A + visual_names_middle

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_Sal', 'G_Sal_Old']
        else:  # during test time, only load Gs
            self.model_names = ['G_Sal']

        # G_Real 生成真实图像
        self.netG_Sal = networks.define_G(opt.input_nc, 1,
                                          64, 'resnet_9blocks', opt.norm, not opt.no_dropout, opt.init_type,
                                          gpu_ids=self.gpu_ids)

        self.netG_Sal_Old = networks.define_G(opt.input_nc, 1,
                                              64, 'resnet_9blocks', opt.norm, not opt.no_dropout, opt.init_type,
                                              gpu_ids=self.gpu_ids)

        # self.CRF_Moudle = ConvCRF.ConvCRF_Cell(opt)

        if self.isTrain:
            self.criterionCycle = torch.nn.L1Loss()

            para = pram_gard(self.netG_Sal)
            self.optimizer_G = torch.optim.Adam(itertools.chain(para),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))  # alpha=0.9 Adam betas=(opt.beta1, 0.999)

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))  # 确定优化策略，多少个step更改学习率

        # if not self.isTrain or opt.continue_train:
        # self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def refresh(self):
        # print('refresh para')
        new_dict = self.netG_Sal.state_dict()
        old_dict = self.netG_Sal_Old.state_dict()
        for i in old_dict:
            old_dict[i] = new_dict[i]

    def set_input(self, input):
        if self.isTrain:
            img = input['img']
            if len(self.gpu_ids) > 0:
                img = img.cuda(self.gpu_ids[0], async=True)
            self.input_img = img
        else:
            img = input['img']
            if len(self.gpu_ids) > 0:
                img = img.cuda(self.gpu_ids[0], async=True)
            self.input_img = img
            self.img_path = input['img_path']

    def set_mode(self, train):
        self.isTrain = train

    def get_image_paths(self):
        return self.img_path

    def forward(self):
        self.input_img = Variable(self.input_img)

    def test(self):
        self.forward()
        self.sal = self.netG_Sal(self.input_img)
        tmp = (self.sal + 1.0) / 2.0
        input = tmp.repeat(1, 3, 1, 1) * ((self.input_img + 1) / 2.0)
        input = input * 2.0 - 1
        self.look = input

    def get_sal_map(self):
        return self.sal

    def backward_G(self):
        # 模型前向传播
        self.sal = self.netG_Sal(self.input_img)
        self.sal_old = self.netG_Sal_Old(self.input_img)
        self.CRF_Out = self.CRF_Moudle.forward(self.input_img, self.sal_old.detach())
        tmp = (self.CRF_Out.detach() + 1.0) / 2.0
        input = tmp.repeat(1, 3, 1, 1) * ((self.input_img.detach() + 1) / 2.0)
        input = input * 2.0 - 1
        self.CRF_Out_look = input
        # self.CRF_Out = self.CRF_Out[:, 0, ...].unsqueeze(0)
        tmp = (self.sal + 1.0) / 2.0
        input = tmp.repeat(1, 3, 1, 1) * self.input_img
        self.look = input

        if not (self.CRF_Out.mean() <= -0.90 or self.CRF_Out.mean() >= 0.90):
            self.loss_Cycle = self.criterionCycle(self.sal, self.CRF_Out.detach())  # 利用CRF
            self.loss_Cycle.backward()

    def optimize_parameters(self, flag_shuffer=False, use=False):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
