#coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    return init_func


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
            witch_moudle = 'vgg',pretrained = True,gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_9blocks_combine':
        netG = ResnetGenerator_Combine(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'fcn':
        netG = FcnGenerator(input_nc, output_nc,witch_moudle, pretrained)
        netG.cuda(gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
        return netG #init_net(netD, init_type, gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def define_C(input_nc, ndf, class_num, norm='batch', init_type='normal', gpu_ids=[],witch_D = 'resnet18',pretrain = False):

    norm_layer = get_norm_layer(norm_type=norm)
    # (self, input_nc, out_nc,witch_moudle,pretrain = False)
    if witch_D =='basic':
        netD = ClassDiscriminator(input_nc, class_num, ndf, n_layers=15, norm_layer=norm_layer)
        return init_net(netD, init_type, gpu_ids)
    else:
        netD = ClassDiscriminator_Resnet(input_nc, class_num,witch_D , pretrain)
        netD.cuda(gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
        return netD #init_net(netD, init_type, gpu_ids)

##############################################################################
# Classes
##############################################################################

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor,l1use=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        if l1use == True:
            self.loss = nn.L1Loss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# 结合分类网络的特征生成噪声 先使用 64 Chinnle的.
class ResnetGenerator_Combine(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_Combine, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        self.model_top = model

        model = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult+64, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_middle = model
        # 把分类的feature_map在这级联要不改了ngf会有问题.64
        model = []
        for i in range(n_downsampling):
            if i == 0:
                other = 64
            else:
                other = 0
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult+other, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model_bottom = model

        self.model_bottom = nn.Sequential(*self.model_bottom)
        self.model_middle = nn.Sequential(*self.model_middle)
        self.model_top = nn.Sequential(*self.model_top)

    def forward(self, input,feature):
        x = self.model_top(input) # 512,56,56
        x = torch.cat([x, feature], 1)
        x = self.model_middle(x) # 512,56,56
        x = self.model_bottom(x) #1, 224, 224
        return x

# 使用FCN 结构
class FcnGenerator(nn.Module):
    def __init__(self, input_nc, out_nc, witch_moudle, pretrained = False):
        super(FcnGenerator, self).__init__()
        self.input_dim = input_nc
        self.out_dim = out_nc
        self.witch_moudle = witch_moudle
        if witch_moudle == 'resnet':
            if pretrained == True:
                model = models.resnet34(pretrained=True)
                self.basic = model
                print('build model with pertrain')
                self.define_module(model,input_nc,witch_moudle)
                self._initialize_weights(True)
            else:
                print('build model with random init')
                model = models.resnet34(pretrained=False)
                self.basic = model
                self.define_module(model,input_nc,witch_moudle)
                self._initialize_weights(False)
        elif witch_moudle == 'vgg':
            if pretrained == True:
                model = models.vgg16_bn(pretrained=True)
                self.basic = model
                print('build model with pertrain')
                self.define_module(model,input_nc,witch_moudle)
                self._initialize_weights(True)
            else:
                print('build model with random init')
                model = models.vgg16_bn(pretrained=False)
                self.basic = model
                self.define_module(model,input_nc,witch_moudle)
                self._initialize_weights(False)

    def _initialize_weights(self,pretrained = False):
        if pretrained == False:
            for m in self.modules():
                for i in m.parameters():
                    nn.init.normal_(i, std=0.01,mean=0)
                    i.requires_grad = True
        else:
            for i in self.basic.parameters():
                i.requires_grad = False
            for m in self.modules():
                for i in m.parameters():
                    if i.requires_grad == False:
                        nn.init.normal_(i, std=0.01,mean=0)



    def define_module(self, model,input_nc,witch_moudle):
        if witch_moudle == 'resnet':
            self.basic_1 = [nn.Conv2d(input_nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)]
            self.basic_1 += [model.bn1]
            self.basic_1 += [model.relu]
            self.basic_1 += [model.maxpool]
            self.basic_1 += [model.layer1] # 56
            self.basic_2 = [model.layer2] # 28
            self.basic_3 = [model.layer3] # 14
            self.basic_4 = [model.layer4] # 7
            self.basic_1 = nn.Sequential(*self.basic_1)
            self.basic_2 = nn.Sequential(*self.basic_2)
            self.basic_3 = nn.Sequential(*self.basic_3)
            self.basic_4 = nn.Sequential(*self.basic_4)
            # 结合部分
            self.score_1 = [nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.score_1 += [nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_1 += [nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_1 += [nn.LeakyReLU(0.2, True)]

            self.score_2 = [nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.score_2 += [nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_2 += [nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_2 += [nn.LeakyReLU(0.2, True)]

            self.score_3 = [nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.score_3 += [nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_3 += [nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_3 += [nn.LeakyReLU(0.2, True)]

            self.score_4 = [nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.score_4 += [nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_4 += [nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_4 += [nn.LeakyReLU(0.2, True)]

            self.score_1 = nn.Sequential(*self.score_1)
            self.score_2 = nn.Sequential(*self.score_2)
            self.score_3 = nn.Sequential(*self.score_3)
            self.score_4 = nn.Sequential(*self.score_4)

            self.upscore_4 = [nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_4 += [nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_4 += [nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.upscore_4 +=[nn.Tanh()]

            self.upscore_43 = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_43 += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_43 += [nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.upscore_43 +=[nn.Tanh()]

            self.upscore_32 = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_32 += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_32 += [nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.upscore_32 +=[nn.Tanh()]

            self.upscore_21 = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_21 += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_21 += [nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.upscore_21 +=[nn.Tanh()]

            self.upscore_last = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_last += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_last += [nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.upscore_last +=[nn.Tanh()]

            self.upscore_4 = nn.Sequential(*self.upscore_4)
            self.upscore_43 = nn.Sequential(*self.upscore_43)
            self.upscore_32 = nn.Sequential(*self.upscore_32)
            self.upscore_21 = nn.Sequential(*self.upscore_21)
            self.upscore_last = nn.Sequential(*self.upscore_last)
        else:
            # VGG Net 做底层特征提取
            self.basic_3 = [nn.Conv2d(input_nc, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            length_model = model.features.__len__()
            model_all = [model.features.__getitem__(i) for i in range(length_model)]
            model_all.pop(0)
            all = []
            index = 0
            flag = 0
            for j in range(length_model):
                tmp = model_all[index:]
                tmp_moudel = []
                if flag == 1:
                    tmp_moudel.append(model_all[index - 1])
                    flag = 0
                if tmp.__len__() == 0:
                    break
                for i in range(tmp.__len__()):
                    if tmp[i]._get_name() == 'MaxPool2d':
                        index += i + 1
                        flag = 1
                        break
                    tmp_moudel.append(tmp[i])
                all.append(tmp_moudel)
            self.basic_3 += all[0]
            self.basic_3 += all[1]
            self.basic_3 += all[2]
            self.basic_4 = all[3]
            self.basic_5 = all[4]
            self.basic_5.pop(-1) # 去掉最后一层的池化层。
            self.basic_3 = nn.Sequential(*self.basic_3)
            self.basic_4 = nn.Sequential(*self.basic_4)
            self.basic_5 = nn.Sequential(*self.basic_5)

            self.score_3 = [nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_3 += [nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_3 += [nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_3 += [nn.LeakyReLU(0.2, True)]

            self.score_4 = [nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_4 += [nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_4 += [nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_4 += [nn.LeakyReLU(0.2, True)]

            self.score_5 = [nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_5 += [nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.score_5 += [nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.score_5 += [nn.LeakyReLU(0.2, True)]

            self.score_3 = nn.Sequential(*self.score_3)
            self.score_4 = nn.Sequential(*self.score_4)
            self.score_5 = nn.Sequential(*self.score_5)

            self.upscore_5 = [nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_5 += [nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_5 +=[nn.Tanh()]

            self.upscore_54 = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_54 += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_54 += [nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
            self.upscore_54 +=[nn.Tanh()]

            self.upscore_43 = [nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_43 += [nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_43 += [nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.upscore_43 +=[nn.Tanh()]

            self.upscore_last = [nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)]
            self.upscore_last += [nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
            self.upscore_last += [nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
            self.upscore_last +=[nn.Tanh()]

            self.upscore_5 = nn.Sequential(*self.upscore_5)
            self.upscore_54 = nn.Sequential(*self.upscore_54)
            self.upscore_43 = nn.Sequential(*self.upscore_43)
            self.upscore_last = nn.Sequential(*self.upscore_last)

    def forward(self, x):
        feature = None
        x = x.clone() #3,224,224
        x[:, 0] = (x[:, 0] * 0.5 + 0.5 - 0.485) / 0.229
        x[:, 1] = (x[:, 1] * 0.5 + 0.5 - 0.456) / 0.224
        x[:, 2] = (x[:, 2] * 0.5 + 0.5 - 0.406) / 0.225
        if self.witch_moudle == 'resnet':
            x1 = self.basic_1(x)
            x2 = self.basic_2(x1)
            x3 = self.basic_3(x2)
            x4 = self.basic_4(x3)
            x1 = self.score_1(x1)
            x2 = self.score_2(x2)
            x3 = self.score_3(x3)
            x4 = self.score_4(x4)
            x4 = self.upscore_4(x4)
            input = torch.cat([x3, x4], 1)
            out = self.upscore_43(input)
            input = torch.cat([x2, out], 1)
            out = self.upscore_32(input)
            input = torch.cat([x1, out], 1)
            out = self.upscore_21(input)
            input = out
            out = self.upscore_last(input)
        else :
            x3 = self.basic_3(x)
            x4 = self.basic_4(x3)
            x5 = self.basic_5(x4)
            x3 = self.score_3(x3)
            x4 = self.score_4(x4)
            x5 = self.score_5(x5)
            x5 = self.upscore_5(x5)

            input = torch.cat([x4, x5], 1)
            out = self.upscore_54(input)
            input = torch.cat([x3, out], 1)
            out = self.upscore_43(input)
            input = out
            out = self.upscore_last(input)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # 最后输出是1？一张feature map？
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

# 分类网络
import torch.nn.functional as F
class ClassDiscriminator(nn.Module):
    def __init__(self, input_nc,class_num,ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(ClassDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        kw = 3
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # 最后输出是1？一张feature map？
        # class_num
        # if use_sigmoid:
        #     sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
        self.fc = nn.Linear(in_features=900, out_features=class_num,bias=True)
        self.fc_tf = nn.Linear(in_features=900, out_features=1, bias=True)

    def forward(self, input):
        input = nn.Upsample(size=(256, 256), mode='bilinear')(input)
        model_out = self.model(input)
        model_out = model_out.view(model_out.size(0), -1)
        label_out = self.fc(model_out)
        tf_out = self.fc_tf(model_out)
        tf_out = F.sigmoid(tf_out)
        label_out = F.sigmoid(label_out)

        return label_out,tf_out


class ClassDiscriminator_Resnet(nn.Module):
    def __init__(self, input_nc, out_nc, witch_moudle, pretrained = False):
        super(ClassDiscriminator_Resnet, self).__init__()
        self.input_dim = input_nc
        self.out_dim = out_nc
        if witch_moudle == 'resnet18':
            if pretrained == True:
                model = models.resnet18(pretrained=True)
                self.basic = model
                print('build model with pertrain')
                self.define_module(model,input_nc)
                self._initialize_weights(True)
            else:
                print('build model with random init')
                model = models.resnet18(pretrained=False)
                self.basic = model
                self.define_module(model,input_nc)
                self._initialize_weights(False)
        elif witch_moudle == 'resnet34':
            if pretrained == True:
                model = models.resnet34(pretrained=True)
                self.basic = model
                print('build model with pertrain')
                self.define_module(model,input_nc)
                self._initialize_weights(True)
            else:
                print('build model with random init')
                model = models.resnet34(pretrained=False)
                self.basic = model
                self.define_module(model,input_nc)
                self._initialize_weights(False)

    def _initialize_weights(self,pretrained = False):
        if pretrained == False:
            for m in self.modules():
                for i in m.parameters():
                    nn.init.normal_(i, std=0.01,mean=0)
                    i.requires_grad = True
        else:
            for i in self.basic.parameters():
                i.requires_grad = True

            for i in self.avgpool.parameters():
                nn.init.normal_(i, std=0.01, mean=0)
                i.requires_grad = True

            for i in self.fc.parameters():
                nn.init.normal_(i, std=0.01, mean=0)
                i.requires_grad = True

    def define_module(self, model,input_nc):

        self.basic_1 = [nn.Conv2d(input_nc, 63, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)]
        self.basic_1_2 = [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        self.basic_2 = [model.relu]
        self.basic_3 = [model.maxpool]
        self.basic_4 = [model.layer1]
        self.basic_5 = [model.layer2]
        self.basic_6 = [model.layer3]
        self.basic_7 = [model.layer4]# 7*7*512
        self.basic_1 = nn.Sequential(*self.basic_1)
        self.basic_1_2 = nn.Sequential(*self.basic_1_2)
        self.basic_2 = nn.Sequential(*self.basic_2)
        self.basic_3 = nn.Sequential(*self.basic_3)
        self.basic_4 = nn.Sequential(*self.basic_4)
        self.basic_5 = nn.Sequential(*self.basic_5)
        self.basic_6 = nn.Sequential(*self.basic_6)
        self.basic_7 = nn.Sequential(*self.basic_7)
        # 旁路结构
        self.bypass = [nn.Conv2d(512, 256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)]
        self.bypass += [nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        self.bypass = nn.Sequential(*self.bypass)
        self.bypass_cls = [nn.ReLU(True)]
        self.bypass_cls += [nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        self.bypass_cls += [nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        self.bypass_cls += [nn.AvgPool2d(kernel_size=7, stride=1, padding=0)]
        self.bypass_cls_fc = [nn.Linear(in_features=128, out_features=self.out_dim, bias=True)]
        self.bypass_cls = nn.Sequential(*self.bypass_cls)
        self.bypass_cls_fc = nn.Sequential(*self.bypass_cls_fc)
        # 分类
        seq_classfiy = [nn.AvgPool2d(kernel_size=7, stride=1, padding=0)]
        # seq_classfiy += [nn.Linear(in_features=512, out_features=21, bias=True)]
        self.avgpool = nn.Sequential(*seq_classfiy)
        seq_classfiy = [nn.Linear(in_features=512, out_features=self.out_dim, bias=True)]
        self.fc = nn.Sequential(*seq_classfiy)

    def forward(self, x,sal_map):
        feature = None
        x = x.clone() #3,224,224
        x[:, 0] = (x[:, 0] * 0.5 + 0.5 - 0.485) / 0.229
        x[:, 1] = (x[:, 1] * 0.5 + 0.5 - 0.456) / 0.224
        x[:, 2] = (x[:, 2] * 0.5 + 0.5 - 0.406) / 0.225

        # x = nn.Upsample(size=(224, 224), mode='bilinear')(x)
        # x = self.basic(x) # 256,7,7
        x = self.basic_1(x)# 64,112,112
        sal_map = self.basic_3(sal_map)
        x = torch.cat([sal_map, x], 1)
        x = self.basic_1_2(x)
        x = self.basic_2(x)# 64,112,112
        x = self.basic_3(x)# 64, 56, 56
        x = self.basic_4(x)# 64, 56, 56
        feature = x
        x = self.basic_5(x)# 128, 28, 28
        x = self.basic_6(x)# 128, 14, 14
        x = self.basic_7(x)# 128, 7, 7
        bypass_feature = self.bypass(x)
        label_new = self.bypass_cls(bypass_feature)
        label_new = label_new.view(x.size(0), -1)
        label_new = self.bypass_cls_fc(label_new)
        label_new = label_new.view(label_new.size(0), -1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        label_out = self.fc(x)
        label_out = label_out.view(label_out.size(0), -1)
        # label_out = F.softmax(label_out,1)

        return label_out,feature,label_new,bypass_feature

    def global_average_pooling(self,x):
        width = x.shape[-1]
        height = x.shape[-2]
        pool_size = [width, height]
        return nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=0)(x)