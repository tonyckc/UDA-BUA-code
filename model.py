'''
@Description:
@Author: Kecheng CHEN
'''
'''
Coral: @ref:https://github.com/facebookresearch/DomainBed/tree/main/domainbed
'''
import random
import functools
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_function import SA, Conv_3d, OutConv,Conv_2d,Conv_2d_trans,Conv_2d_padding
from model_function import AE_Down, AE_Up
from torch import nn
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.variational_layers.conv_variational import ConvTranspose2dReparameterization
from utils import updata_ae
import torch
import numpy as np
import time
from torchvision import models
from collections import OrderedDict
'''
ERM
'''

def mlv(fm):
    [batch,c, h, w] = fm.shape

    xs = h
    ys = w
    x = fm
    x1 = torch.zeros_like(x)
    x2 = torch.zeros_like(x)
    x3 = torch.zeros_like(x)
    x4 = torch.zeros_like(x)
    x5 = torch.zeros_like(x)
    x6 = torch.zeros_like(x)
    x7 = torch.zeros_like(x)
    x8 = torch.zeros_like(x)
    x9 = torch.zeros_like(x)

    x1[:,:,1: xs - 2, 1: ys - 2] = x[:,:,2: xs - 1, 2: ys - 1]
    x2[:,:,1: xs - 2, 2: ys - 1] = x[:,:,2: xs - 1, 2: ys - 1]
    x3[:,:,1: xs - 2, 3: ys]  = x[:,:,2: xs - 1, 2: ys - 1]
    x4[:,:,2: xs - 1, 1: ys - 2] = x[:,:,2: xs - 1, 2: ys - 1]
    x5[:,:,2: xs - 1, 2: ys - 1] = x[:,:,2: xs - 1, 2: ys - 1]
    x6[:,:,2: xs - 1, 3: ys]  = x[:,:,2: xs - 1, 2: ys - 1]
    x7[:,:,3: xs, 1: ys - 2]   = x[:,:,2: xs - 1, 2: ys - 1]
    x8[:,:,3: xs, 2: ys - 1]   = x[:,:,2: xs - 1, 2: ys - 1]
    x9[:,:,3: xs, 3: ys]     = x[:,:,2: xs - 1, 2: ys - 1]

    x1 = x1[:,:,2:xs - 1, 2: ys - 1]
    x2 = x2[:,:,2:xs - 1, 2: ys - 1]
    x3 = x3[:,:,2:xs - 1, 2: ys - 1]
    x4 = x4[:,:,2:xs - 1, 2: ys - 1]
    x5 = x5[:,:,2:xs - 1, 2: ys - 1]
    x6 = x6[:,:,2:xs - 1, 2: ys - 1]
    x7 = x7[:,:,2:xs - 1, 2: ys - 1]
    x8 = x8[:,:,2:xs - 1, 2: ys - 1]
    x9 = x9[:,:,2:xs - 1, 2: ys - 1]

    d1 = torch.abs(x1 - x5)
    d2 = torch.abs(x2 - x5)
    d3 = torch.abs(x3 - x5)
    d4 = torch.abs(x4 - x5)
    d5 = torch.abs(x6 - x5)
    d6 = torch.abs(x7 - x5)
    d7 = torch.abs(x8 - x5)
    d8 = torch.abs(x9 - x5)

    dd = torch.max(d1, d2)
    dd = torch.max(dd, d3)
    dd = torch.max(dd, d4)
    dd = torch.max(dd, d5)
    dd = torch.max(dd, d6)
    dd = torch.max(dd, d7)
    dd = torch.max(dd, d8)

    map = dd

    return map





class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self):
        super(Algorithm, self).__init__()


    def update(self, xt,xs,y, batch=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
"""
Generator
"""
##******************************************************************************************************************************
class SACNN(nn.Module):
    def __init__(self, N, version):
        super(SACNN, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.N = N

        if version is "v2":
            self.lay1 = Conv_3d(in_ch=self.input_channels, out_ch=64, use_relu="use_relu")
            self.lay2 = Conv_3d(in_ch=64, out_ch=32, use_relu="use_relu")
            # self.lay3 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay3 = Conv_3d(in_ch=32, out_ch=32, use_relu="use_relu")
            self.lay4 = Conv_3d(in_ch=32, out_ch=16, use_relu="use_relu")
            # self.lay5 = SA(in_ch=2, out_ch=2, N=self.N)
            self.lay5 = Conv_3d(in_ch=16, out_ch=16, use_relu="use_relu")
            self.lay6 = Conv_3d(in_ch=16, out_ch=32, use_relu="use_relu")
            # self.lay7 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay7 = Conv_3d(in_ch=32, out_ch=32, use_relu="use_relu")
            self.lay8 = Conv_3d(in_ch=32, out_ch=64, use_relu="use_relu")
            self.lay9 = OutConv(in_ch=64, out_ch=self.output_channels)
        elif version is "v1":
            self.lay1 = Conv_3d(in_ch=self.input_channels, out_ch=8, use_relu="use_relu")
            self.lay2 = Conv_3d(in_ch=8, out_ch=4, use_relu="use_relu")
            # self.lay3 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay3 = Conv_3d(in_ch=4, out_ch=4, use_relu="use_relu")
            self.lay4 = Conv_3d(in_ch=4, out_ch=2, use_relu="use_relu")
            # self.lay5 = SA(in_ch=2, out_ch=2, N=self.N)
            self.lay5 = Conv_3d(in_ch=2, out_ch=2, use_relu="use_relu")
            self.lay6 = Conv_3d(in_ch=2, out_ch=4, use_relu="use_relu")
            # self.lay7 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay7 = Conv_3d(in_ch=4, out_ch=4, use_relu="use_relu")
            self.lay8 = Conv_3d(in_ch=4, out_ch=8, use_relu="use_relu")
            self.lay9 = OutConv(in_ch=8, out_ch=self.output_channels)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)
        x = self.lay8(x)
        x = self.lay9(x)
        return x

class CPCE(nn.Module):
    def __init__(self):
        super(CPCE, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.layer1 = Conv_2d(in_ch=self.input_channels,out_ch=32,use_relu="None")
        self.layer2 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer3 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer4 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")

        self.layer5 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer5_2 = Conv_2d(in_ch=64,out_ch=32,use_relu="use_relu",kernel=1)
        self.layer6 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer6_2 = Conv_2d(in_ch=64, out_ch=32, use_relu="use_relu", kernel=1)
        self.layer7 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer7_2 = Conv_2d(in_ch=64, out_ch=32, use_relu="use_relu", kernel=1)
        self.layer8 = Conv_2d_trans(in_ch=32, out_ch=1, use_relu="use_relu")
        self.nn = nn.Sequential(self.layer1,self.layer2,self.layer3,self.layer4,self.layer5,self.layer6,self.layer7,self.layer8)
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = F.relu(x1)
        x2 = self.layer2(x2)
        x3 = F.relu(x2)

        x3 = self.layer3(x3)
        x4 = F.relu(x3)

        x4 = self.layer4(x4)
        encoder = F.relu(x4)

        x5 = self.layer5(encoder)
        x5 = torch.concat((x3,x5),dim=1)
        x6 = F.relu(x5)
        x6 = self.layer5_2(x5)

        x6 = self.layer6(x6)
        x6 = torch.concat((x6,x2),dim=1)
        x7 = F.relu(x6)
        x7 = self.layer6_2(x6)

        x7 = self.layer7(x7)
        x7 = torch.concat((x7,x1),dim=1)
        x8 = F.relu(x7)
        x8 = self.layer7_2(x7)

        output = self.layer8(x8)
        return output#, encoder

        #return self.nn(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.layer1 = Conv_2d(in_ch=self.input_channels,out_ch=32,use_relu="None")
        self.layer2 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer3 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer4 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer5 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = F.relu(x1)
        x2 = self.layer2(x2)
        x3 = F.relu(x2)
        x3 = self.layer3(x3)
        x4 = F.relu(x3)
        x4 = self.layer4(x4)
        x5 = F.relu(x4)
        x5 = self.layer5(x5)
        x6 = F.relu(x5)

        return x6

class Encoder_part1(nn.Module):
    def __init__(self):
        super(Encoder_part1, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.layer1 = Conv_2d(in_ch=self.input_channels,out_ch=32,use_relu="None")
        self.layer2 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer3 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        self.layer4 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")
        #self.layer5 = Conv_2d(in_ch=32, out_ch=32, use_relu="None")

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = F.relu(x1)
        x2 = self.layer2(x2)
        x3 = F.relu(x2)
        x3 = self.layer3(x3)
        x4 = F.relu(x3)
        x4 = self.layer4(x4)
        x5 = F.relu(x4)
        #x5 = self.layer5(x5)
        #x6 = F.relu(x5)

        return x5#6

class Decoder_bayesian(nn.Module):
    def __init__(self):
        super(Decoder_bayesian, self).__init__()
        self.input_channels = 32
        self.output_channels = 32
        self.layer5 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer6 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer7 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer8 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        

    def forward(self, x):

        encoder = x
        x5 = self.layer5(encoder)
        x6 = F.relu(x5)
        x6 = self.layer6(x6)
        x7 = F.relu(x6)
        x7 = self.layer7(x7)
        x8 = F.relu(x7)
        x8 = self.layer8(x8)
        x9 = F.relu(x8)
        output = x9
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_channels = 32
        self.output_channels = 32

        self.layer5 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer6 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer7 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer8 = Conv_2d_trans(in_ch=32, out_ch=32, use_relu="None")
        self.layer9 = Conv_2d_trans(in_ch=32, out_ch=1, use_relu="use_relu")

    def forward(self, x):

        encoder = x
        x5 = self.layer5(encoder)
        x6 = F.relu(x5)
        x6 = self.layer6(x6)
        x7 = F.relu(x6)
        x7 = self.layer7(x7)
        x8 = F.relu(x7)
        x8 = self.layer8(x8)
        x9 = F.relu(x8)

        output = self.layer9(x9)
        return output
'''
ERM
'''
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self):
        super(ERM, self).__init__()

    def update(self, xt,xs,y, batch=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
'''
MMD
'''
class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, gaussian,estimate):
        super(AbstractMMD, self).__init__()
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"
        self.estimate = estimate
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()
            #print(mean_diff + cova_diff)
            #exit()
            return mean_diff + cova_diff

    def compute(self,zs):
        batch_size, num_mc, c, w, h = zs.shape

        zs = torch.reshape(zs, (batch_size, num_mc, c * w * h))
        total_coral_loss = 0
        for i in range(batch_size):
                index_1 = i
                x = torch.squeeze(zs[index_1])
                mean_x = x.mean(0, keepdim=True)
                cent_x = x - mean_x
                cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)

                mean_diff = (mean_x).pow(2).mean()
                cova_diff = (cova_x).pow(2).mean()

                total_coral_loss += (cova_diff+mean_diff)

        loss = total_coral_loss / (batch_size)
        return  loss

    def update(self, zs,zt,device=None):
        # zs shape: batch_size, num_mc, channle, weight, height
        # zt shape: both
        batch_size,num_mc,c,w,h = zs.shape

        zs = torch.reshape(zs,(batch_size,num_mc,c*w*h))
        zt = torch.reshape(zt,(batch_size,num_mc,c*w*h))

        total_coral_loss = 0
        if self.estimate:
            m = int(batch_size / 2)
            for i in range(m):
                    index_1 = 2 * i
                    index_2 = (2 * i)+1
                    total_coral_loss += self.mmd(torch.squeeze(zs[index_1]),torch.squeeze(zt[index_2]))
            loss = total_coral_loss/m
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    index_1 = i
                    index_2 = j
                    total_coral_loss += self.mmd(torch.squeeze(zs[index_1]), torch.squeeze(zt[index_2]))
            loss = total_coral_loss / (batch_size*batch_size)
        return loss



'''
CORAL
'''
class Bayesian_CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """
    def __init__(self,gaussian,estimate):
        super(Bayesian_CORAL, self).__init__(gaussian=False,estimate=True)

"""
Perceptual loss
"""
##******************************************************************************************************************************
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.lay1 = AE_Down(in_channels=1, out_channels=64)
        self.lay2 = AE_Down(in_channels=64, out_channels=128)
        self.lay3 = AE_Down(in_channels=128, out_channels=256)
        self.lay4 = AE_Down(in_channels=256, out_channels=256)

        self.lay5 = AE_Up(in_channels=256, out_channels=256)
        self.lay6 = AE_Up(in_channels=256, out_channels=128)
        self.lay7 = AE_Up(in_channels=128, out_channels=64)
        self.lay8 = AE_Up(in_channels=64, out_channels=32)
        self.lay9 = OutConv(in_ch=32, out_ch=1)

        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.deconv1 = nn.ConvTranspose3d(128, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.deconv2 = nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        x = self.lay1(x)
        x = self.maxpool(x)
        x = self.lay2(x)
        x = self.maxpool(x)
        x = self.lay3(x)
        y = self.lay4(x)

        x = self.lay5(y)
        x = self.lay6(x)
        x = self.deconv1(x)
        x = self.lay7(x)
        x = self.deconv2(x)
        x = self.lay8(x)
        out = self.lay9(x)
        return out, y


class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False,remote=False):
        super(Vgg19_out, self).__init__()
        print('-'*20,'loading VGG Model...','-'*20)
        vgg = models.vgg19(pretrained=False) # .cuda()
        if remote:
            vgg.load_state_dict(torch.load('./pretrained_model/vgg19_new.pth'))#/media/ubuntu/88692f9c-d324-4895-8764-1cf202f9e6ac/chenkecheng
        else:
            vgg.load_state_dict(torch.load('./pretrained_model/vgg19_new.pth'))
        print('-' * 20, 'loading VGG Model Done', '-' * 20)
        vgg.eval()
        vgg_pretrained_features = vgg.features
        #print(vgg.state_dict()['features.0.bias'])
        #exit()
        self.requires_grad = requires_grad
        #self.slice1 = torch.nn.Sequential()
        #self.slice2 = torch.nn.Sequential()
        #self.slice3 = torch.nn.Sequential()
        #self.slice4 = torch.nn.Sequential()
        #self.slice5 = torch.nn.Sequential()

        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36])
        #print(self.feature_extractor)
        #exit()
        #for x in range(4, 9):  # (3, 7):
        #    self.slice2.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(9, 14):  # (7, 12):
        #    self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(14, 23):  # (12, 21):
        #    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(23, 32):  # (21, 30):
        #    self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        X = torch.concat((inputs * 255 - 103.939, inputs * 255 - 116.779, inputs * 255 - 123.68), dim=1)
        #X = torch.concat((inputs, inputs, inputs), dim=1)
        h_relu1 = self.feature_extractor(X)
        #h_relu2 = self.slice2(h_relu1)
        #h_relu3 = self.slice3(h_relu2)
        #h_relu4 = self.slice4(h_relu3)
        #h_relu5 = self.slice5(h_relu4)
        out = h_relu1#[h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self,remote=False):

        super(VGGLoss, self).__init__()
        self.vgg = Vgg19_out(requires_grad=False,remote=remote)
        #self.criterion = nn.MSELoss(reduction='sum')
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]


    def forward(self, x, y,batch_size):
        #print(x.shape)
        #exit()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        #print(x_vgg.shape)
        #exit()
        #for iter, (x_fea, y_fea) in enumerate(zip(x_vgg, y_vgg)):
            #print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
        loss = torch.sum(torch.square(x_vgg - y_vgg))/(batch_size*4*4*512)
        #print(loss)
        #exit()
        return  loss


"""
Discriminator
"""
##******************************************************************************************************************************

class Discriminator_diy(nn.Module):
    def __init__(self):
        super(Discriminator_diy, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 15 * 15, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class DISC(nn.Module):
    def __init__(self):
        super(DISC, self).__init__()

        self.lay1 = Conv_2d_padding(in_ch=1, out_ch=16, use_relu="use_relu")
        self.lay2 = Conv_2d_padding(in_ch=16, out_ch=32, use_relu="use_relu")
        self.lay3 = Conv_2d_padding(in_ch=32, out_ch=64, use_relu="use_relu")
        self.lay4 = Conv_2d_padding(in_ch=64, out_ch=64, use_relu="use_relu")
        self.lay5 = Conv_2d_padding(in_ch=64, out_ch=32, use_relu="use_relu")
        self.lay6 = Conv_2d_padding(in_ch=32, out_ch=16, use_relu="use_relu")
        self.lay7 = Conv_2d_padding(in_ch=16, out_ch=1, use_relu="use_relu")

        ## out.view(-1, 256*self.output_size*self.output_size)
        self.fc1 = nn.Sequential(nn.Linear(1*64*64, 1024),nn.LeakyReLU(0.2))    ## input:N*C*D*H*W=N*1*3*64*64
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)
        #print(x.shape)
        x = x.view(-1, 256 * 256)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class WGAN(nn.Module):
    def __init__(self, args, device = None):
        super(WGAN, self).__init__()
        self.device = device
        self.generator = CPCE()
        self.discriminator = DISC()
        self.p_criterion = VGGLoss(remote=args.remote)
        if args.MAE:
            self.mse = nn.L1Loss()
        else:
            self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        self.lamda_style = args.style_lambda
        #print(self.lamda_p)
        #print(self.gamma_m)
        #exit()
    def updata(self,x,y,batch_size):
        prediction, embedding_z = self.generator.forward(x)
        #mse_loss = self.mse(prediction,y)
        g_loss = self.get_g_loss(x)
        per_loss = self.p_criterion.forward(prediction,y,batch_size)
        #per_loss = per_loss.to(dtype=x.dtype)
        #per_loss = per_loss.to(self.device)
        total_loss = g_loss + self.lamda_style*per_loss
        total_loss = total_loss.to(self.device)
        return 0,0,0,g_loss,per_loss,total_loss
    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def get_g_loss(self, x,perceptual=True, return_p=False):
        """
        generator loss
        """
        fake,_ = self.generator.forward(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        return  g_loss
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100


    def discrinator_pass(self,x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake,_ = self.generator.forward(x)
        d_real = self.discriminator(y)

        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return loss, gp_loss


    def predict(self,x):
        output, _ = self.generator.forward(x)
        return output


class BasicBlock(nn.Module):  #
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual  #
        return out


class Generator(nn.Module):  # 


    def __init__(self, layers):
        super(Generator, self).__init__()

        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer([32, 32], layers-2)  # self._make_layer  416,416,32 -> 208,208,64
        self.outplanes = 32
        self.conv9 = nn.Conv2d(self.outplanes, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu9 = nn.ReLU()

    def _make_layer(self, planes, blocks):  # 
        layers = []
        #layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        #layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        #layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))  # 
        return nn.Sequential(OrderedDict(
            layers))  # 

    def forward(self, x):  #
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.conv9(x)
        output = self.relu9(x)

        return output


class GANLoss(nn.Module):
    def __init__(self, device=None,use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.device = device
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False).to(self.device)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False).to(self.device)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class disc(nn.Module):
    def __init__(self, args, device = None):
        super(disc, self).__init__()
        self.device = device

        self.generator = NLayerDiscriminator(1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True)


        self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        self.lamda_style = args.style_lambda
        #print(self.lamda_p)
        #print(self.gamma_m)
        #exit()
    def updata(self,x,y,batch_size):
        prediction= self.generator.forward(x)

        mse_loss = self.mse(prediction,y)


        g_loss = Variable(torch.tensor(0)).cuda()#self.get_g_loss(x)
        per_loss = Variable(torch.tensor(0)).cuda()#self.p_criterion.forward(prediction,y,batch_size)
        #per_loss = per_loss.to(dtype=x.dtype)
        #per_loss = per_loss.to(self.device)
        total_loss = mse_loss#g_loss + self.lamda_style*per_loss
        total_loss = total_loss.to(self.device)
        return Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),g_loss,per_loss,total_loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def get_g_loss(self, x,perceptual=True, return_p=False):
        """
        generator loss
        """
        fake,_ = self.generator.forward(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        return  g_loss
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100


    def discrinator_pass(self,x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake,_ = self.generator.forward(x)
        d_real = self.discriminator(y)

        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return loss, gp_loss


    def predict(self,x):
        output = self.generator.forward(x)
        return output




class UDA(nn.Module):
    def __init__(self, args, device = None):
        super(UDA, self).__init__()
        self.device = device
        self.generator = CPCE()
        self.discriminator = NLayerDiscriminator(1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True)


        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        self.lamda_style = args.style_lambda
        #print(self.lamda_p)
        #print(self.gamma_m)
        #exit()
    def updata(self,x,y,xt,yt,xt_noiser,batch_size):
        prediction_s= self.generator.forward(x)
        prediction_s = self.discriminator(prediction_s)
        prediction_t = self.generator.forward(xt)
        prediction_t = self.discriminator(prediction_t)
        # loss da
        mse_loss_s = self.mse(prediction_s,y)
        mse_loss_t = self.mse(prediction_t,yt)
        mse_loss = mse_loss_s + mse_loss_t
        # loss aug
        prediction_t_noiser = self.generator.forward(xt_noiser)
        loss_aug = self.mae(xt,prediction_t_noiser)


        #g_loss = 0#self.get_g_loss(x)
        #per_loss = 0#self.p_criterion.forward(prediction,y,batch_size)
        #per_loss = per_loss.to(dtype=x.dtype)
        #per_loss = per_loss.to(self.device)
        total_loss = loss_aug + 0.2 * mse_loss#g_loss + self.lamda_style*per_loss
        total_loss = total_loss.to(self.device)
        return mse_loss,mse_loss,mse_loss,loss_aug,mse_loss,total_loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def get_g_loss(self, x,perceptual=True, return_p=False):
        """
        generator loss
        """
        fake = self.generator.forward(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        return  g_loss
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100


    def discrinator_pass(self,x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake = self.generator.forward(x)
        d_real = self.discriminator(y)

        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return loss, gp_loss


    def predict(self,x):
        output = self.generator.forward(x)
        return output




class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
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

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class cyclegan(nn.Module):
    def __init__(self, args, device = None):
        super(cyclegan, self).__init__()
        self.device = device
        self.lambda_idt = 0.75#0.75
        self.lambda_A = 30#20
        self.lambda_B = 30#20
        self.Tensor = torch.Tensor
        self.netG_A = Generator(9)
        self.netG_B = Generator(9)
        self.netD_A = NLayerDiscriminator(1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True)
        self.netD_B = NLayerDiscriminator(1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True)
        self.criterionGAN = GANLoss(device=self.device,use_lsgan=True, tensor=self.Tensor)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        #self.p_criterion = VGGLoss(remote=args.remote)
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)
        #print(self.lamda_p)
        #print(self.gamma_m)
        #exit()

    def updata(self,ys,xt,batch_size):

        # identity loss
        idt_A = self.netG_A(ys)
        loss_idt_A = self.criterionIdt(idt_A, ys) * self.lambda_B * self.lambda_idt
        # G_B should be identity if real_A is fed.
        idt_B = self.netG_B(xt)
        loss_idt_B = self.criterionIdt(idt_B, xt) * self.lambda_A * self.lambda_idt

        #self.idt_A = idt_A.data
        #self.idt_B = idt_B.data
        #self.loss_idt_A = loss_idt_A.data[0]
        #self.loss_idt_B = loss_idt_B.data[0]

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(xt)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(ys)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, xt) * self.lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, ys) * self.lambda_B
        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        return Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),Variable(torch.tensor(0)).cuda(),loss_G

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        return loss_D


    def dis_A(self,ys):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, ys, fake_B)
        #self.loss_D_A = loss_D_A.data[0]
        return loss_D_A

    def dis_B(self, xt):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, xt, fake_A)
        return loss_D_B

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def get_g_loss(self, x,perceptual=True, return_p=False):
        """
        generator loss
        """
        fake,_ = self.generator.forward(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        return  g_loss
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100


    def discrinator_pass(self,x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake,_ = self.generator.forward(x)
        d_real = self.discriminator(y)

        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return loss, gp_loss

    def predict(self,x):
        output = self.netG_A.forward(x)
        return output



class RFM(nn.Module):
    def __init__(self, args, device = None):
        super(RFM, self).__init__()
        self.device = device
        self.generator = CPCE()
        self.discriminator = DISC()
        self.p_criterion = VGGLoss(remote=args.remote)
        if args.MAE:
            self.mse = nn.L1Loss()
        else:
            self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        self.lamda_style = args.style_lambda
        #print(self.lamda_p)
        #print(self.gamma_m)
        #exit()
    def updata(self,x,y,batch_size):
        prediction = self.generator.forward(x)
        mse_loss = self.mse(prediction,y)
        g_loss = self.get_g_loss(x)
        per_loss = self.p_criterion.forward(prediction,y,batch_size)
        #per_loss = per_loss.to(dtype=x.dtype)
        #per_loss = per_loss.to(self.device)
        total_loss = 0.1*g_loss + per_loss + mse_loss
        total_loss = total_loss.to(self.device)
        return per_loss,per_loss,per_loss,g_loss,per_loss,total_loss
    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def get_g_loss(self, x,perceptual=True, return_p=False):
        """
        generator loss
        """
        fake = self.generator.forward(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        return  g_loss
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100


    def discrinator_pass(self,xt,xs, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake = self.generator.forward(xs)
        real = self.generator.forward(xt)
        d_real = self.discriminator(real)

        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(real, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return loss, gp_loss


    def predict(self,x):
        output = self.generator.forward(x)
        return output


class Baseline(nn.Module):
    def __init__(self, args, device=None):
        super(Baseline, self).__init__()
        self.device = device

        self.mc_training = 1
        self.mope = args.mope
        self.mope_delta = args.mope_delta
        self.encoder_denoising = Encoder()
        self.mu = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
        self.posterior_rho_init = args.posterior_rho_init
        self.decoder_denoising_part1 = Decoder_bayesian()
        self.regression = ConvTranspose2dReparameterization(in_channels=32, out_channels=1, kernel_size=3, padding=0,
                                                            posterior_rho_init=self.posterior_rho_init)
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.posterior_rho_init,
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": self.mope,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.mope_delta,  # 0.5=>0.2
        }
        dnn_to_bnn(self.mu, const_bnn_prior_parameters)

        self.generator = nn.Sequential(self.encoder_denoising,self.mu,self.decoder_denoising_part1,self.regression)

        self.p_criterion = VGGLoss(remote=args.remote)
        if args.MAE:
            self.mse = nn.L1Loss()
        else:
            self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        # print(self.lamda_p)
        # print(self.gamma_m)
        # exit()



    def updata(self, x, y, batch_size):
        output_ = []
        kl_ = []
        for mc_run in range(1):
            z_s = self.encoder_denoising(x)
            embedding_zs = self.mu(z_s)
            output_zs = self.decoder_denoising_part1(embedding_zs)
            output_zs, kl_zs = self.regression(output_zs, return_kl=True)
            output_zs = F.relu(output_zs)
            output_.append(output_zs)
            kl = get_kl_loss(self.mu) + kl_zs  # get_kl_loss(self.regression)
            kl_.append(kl)

        prediction = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)

        scaled_kl = kl / batch_size

        mse_loss = self.mse(prediction, y)
        per_loss = self.p_criterion.forward(prediction, y, batch_size)
        # per_loss = per_loss.to(dtype=x.dtype)
        # per_loss = per_loss.to(self.device)
        total_loss = scaled_kl + self.gamma_m * mse_loss + self.lamda_p * per_loss
        total_loss = total_loss.to(self.device)
        return 0, 0, scaled_kl, mse_loss, per_loss, total_loss

    def predict(self, x):
        output_ = []
        for mc_run in range(20):
            output, _ = self.generator.forward(x)
            output_.append(output)
        prediction = torch.mean(torch.stack(output_), dim=0)
        return prediction


class Baseline_deter(nn.Module):
    def __init__(self, args, device=None):
        super(Baseline_deter, self).__init__()
        self.device = device


        self.generator = CPCE()

        self.p_criterion = VGGLoss(remote=args.remote)
        if args.MAE:
            self.mse = nn.L1Loss()
        else:
            self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        # print(self.lamda_p)
        # print(self.gamma_m)
        # exit()



    def updata(self, x, y, batch_size):
        prediction = self.generator(x)

        mse_loss = self.mse(prediction, y)
        per_loss = self.p_criterion.forward(prediction, y, batch_size)
        # per_loss = per_loss.to(dtype=x.dtype)
        # per_loss = per_loss.to(self.device)
        total_loss =  self.gamma_m * mse_loss + self.lamda_p * per_loss
        total_loss = total_loss.to(self.device)
        return mse_loss, per_loss, total_loss, mse_loss, per_loss, total_loss

    def predict(self, x):

        prediction = self.generator.forward(x)
        return prediction


class noiser2noise(nn.Module):
    def __init__(self, args, device=None):
        super(noiser2noise, self).__init__()
        self.device = device
        self.generator = CPCE()
        self.mse = nn.MSELoss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        # print(self.lamda_p)
        # print(self.gamma_m)
        # exit()

    def updata(self, x, y, batch_size):

        prediction = self.generator.forward(x)
        mse_loss = self.mse(prediction, y)
        #per_loss = self.p_criterion.forward(prediction, y, batch_size)
        # per_loss = per_loss.to(dtype=x.dtype)
        # per_loss = per_loss.to(self.device)
        total_loss = mse_loss #+ self.lamda_p * per_loss
        total_loss = total_loss.to(self.device)
        return Variable(torch.tensor(0)).cuda(), Variable(torch.tensor(0)).cuda(), Variable(torch.tensor(0)).cuda(), mse_loss, Variable(torch.tensor(0)).cuda(), total_loss

    def predict(self, x):
        alpha = 1
        output= self.generator.forward(x)
        prediction = ((1 + alpha ** 2)*output - x) / (alpha ** 2)
        return prediction





class RedCNN(nn.Module):
    def __init__(self, args, device=None):
        super(RedCNN, self).__init__()
        self.device = device
        self.generator = CPCE()
        self.mse = nn.L1Loss()
        self.lamda_p = args.p_lambda
        self.gamma_m = args.m_gamma
        # print(self.lamda_p)
        # print(self.gamma_m)
        # exit()

    def updata(self, x, y, batch_size):
        prediction, embedding_z = self.generator.forward(x)
        mse_loss = self.mse(prediction, y)

        total_loss =  mse_loss
        total_loss = total_loss.to(self.device)
        return 0, 0, 0, mse_loss, 0, total_loss

    def predict(self, x):
        output, _ = self.generator.forward(x)
        return output


class DA_Denoiser(nn.Module):
    def __init__(self, args, device = None):
        super(DA_Denoiser, self).__init__()
        self.device = device
        self.direct = args.direct
        self.unpaired = args.unpaired
        self.Tensor = torch.Tensor
        self.gp_bool = args.gp
        self.posterior_rho_init = args.posterior_rho_init
        self.mope = args.mope
        self.coral = args.coral
        self.remote = args.remote
        self.xt_reconstruction_weight = args.xt_reconstruction_weight
        self.style = args.style
        self.xt_reconstruction = args.xt_reconstruction
        self.mope_delta = args.mope_delta
        self.style_lambda = args.style_lambda
        # beta1 in Eq. (14)
        self.source_weight = args.source_weight
        self.bayesian_coral_weight = args.bayesian_coral
        self.num_mc = args.num_mc

        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": self.posterior_rho_init,
            "type": "Reparameterization",  # "Flipout",#"Reparameterization",  # Flipout or Reparameterization
            "moped_enable": self.mope,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": self.mope_delta,  # 0.5=>0.2
        }
        
        # To accelarate the training and inference, only last layer of encoder of the denoising network (generator) is converted into the BNN 
        self.encoder_denoising = Encoder()
        self.mu = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
        # convert the last layer into the BNN 
        dnn_to_bnn(self.mu, const_bnn_prior_parameters)
        
        # Self-recontruction loss related network, refers to Eq. (9)
        if self.xt_reconstruction:
            self.decoder_content = CPCE()
            self.content_mse = nn.MSELoss()

        self.lsgan = args.lsgan
        # if use  Sharpness-aware Distribution Alignment via Adversarial Training, which GAN type is used.
        if self.style:

            if not self.lsgan:
                self.discriminator = DISC()
                self.fake_t_pool = ImagePool(50)
            else:
                self.discriminator =NLayerDiscriminator(1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True)
                self.fake_t_pool = ImagePool(50)
                self.criterionGAN = GANLoss(device=self.device, use_lsgan=True, tensor=self.Tensor)

        # To accelarate the training and inference, only last layer of decoder of the denoising network (generator) is converted into the BNN
        self.decoder_denoising_part1 = Decoder_bayesian()
        self.regression = ConvTranspose2dReparameterization(in_channels=32, out_channels=1, kernel_size=3, padding=0,posterior_rho_init = self.posterior_rho_init)

        

        # perception loss
        self.p_criterion = VGGLoss(remote=args.remote)
        # use MAE or MSE
        self.mse = nn.L1Loss() if args.MAE else nn.MSELoss()
        # weight of perception loss
        self.lamda_p = args.p_lambda

        # use the Bayesian coral, i.e., Bayesian Uncertainty Alignment
        if self.coral == True or args.validate_coral == True:
            self.bayesian_coral = Bayesian_CORAL(gaussian=False,estimate=args.estimate_coral)
            #  global average pooling,
            self.squeeze = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)))
        # overall generator network, decoder_content is the Self-recontruction loss related network
        self.generator = nn.Sequential(self.encoder_denoising,self.mu,self.decoder_denoising_part1,self.regression, self.decoder_content) # encoder_denoising
    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)

        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty


    def updata_v2(self,xt,xs,y,batch_size,epoch=0):
        '''
        In this version, we aim to improve the manner of bayesian coral.
        '''
        # generate determinative features before the BNN layer
        z_t = self.encoder_denoising(xt)
        z_s = self.encoder_denoising(xs)

        embeddings_lists_zs = []
        embeddings_lists_zt = []
        kl_lists = []

        accelerate_training = True
        for mc_run in range(self.num_mc):
            input_zs = z_s
            input_zt = z_t
            # bayesian layer on encoder, sampling from BNN
            embedding_zs = self.mu(input_zs)
            embedding_zt = self.mu(input_zt)
            embeddings_lists_zs.append(embedding_zs)
            embeddings_lists_zt.append(embedding_zt)
            # To accelerate training and inference, the reconstrcted y only pass one time in the decoder side
            if accelerate_training:
                # sample from BNN decoder (only last layer)
                embedding_zt.requires_grad_(True)
                output_zs = self.decoder_denoising_part1(embedding_zs)
                output_zs,kl_zs = self.regression(output_zs,return_kl=True)
                output_zt = self.decoder_denoising_part1(embedding_zt)
                output_zt, kl_zt = self.regression(output_zt, return_kl=True)
                output_zs = F.relu(output_zs)
                output_zt = F.relu(output_zt)

                # self-reconstruction module
                output_zt_re= self.decoder_content(output_zt)
                # kl loss in BNN
                kl = get_kl_loss(self.mu) + kl_zs
                kl_lists.append(kl)
                accelerate_training = False
                
        ys_hat = output_zs
        yt_hat = output_zt

        # L_SDA:  generator loss based on MLV, refers to the second term of Eq. (13) and last term of Eq. (14)
        if self.style:
            yt_hat_mlv = mlv(yt_hat)
            d_fake = self.discriminator(yt_hat_mlv)
            if self.lsgan:
                d_loss = self.criterionGAN(d_fake, True)#
            else:
                d_loss = -torch.mean(d_fake)  #
            loss_style = d_loss
        else:
            loss_style = Variable(torch.tensor(0)).cuda()

        # first two terms of Eq. 15
        loss_xs_denoising = self.mse(ys_hat, y) + self.p_criterion.forward(ys_hat, y, batch_size)

        # self-reconstrcution loss, refers to Eq. (9)
        if self.xt_reconstruction:
            xt = Variable(torch.tensor(xt, requires_grad=True)).cuda()
            xt_reconstruction_loss = torch.mean(torch.square(output_zt_re - xt))
        else:
            xt_reconstruction_loss = Variable(torch.tensor(0)).cuda()


        # kl loss of BNN, get their mean
        kl_loss = torch.mean(torch.stack(kl_lists), dim=0)
        scaled_kl = kl_loss / batch_size

        # bayesian coral, i.e., Bayesian Uncertainty Alignment loss
        if self.coral:
            # global average pooling for each features, refers to Eq. (6)
            embeddings_lists_zt = torch.stack(embeddings_lists_zt)
            embeddings_out_zt = [self.squeeze(embeddings_lists_zt[index]) for index in range(self.num_mc)]
            embeddings_out_zt = torch.stack(embeddings_out_zt)
            embeddings_out_zt = torch.transpose(embeddings_out_zt, 0, 1)
            # shape: batchsize, num_mc, c, w, h
            ndm_zt = embeddings_out_zt

            # global average pooling for each features, refers to Eq. (6)
            embeddings_lists_zs = torch.stack(embeddings_lists_zs)
            embeddings_out_zs = [self.squeeze(embeddings_lists_zs[index]) for index in range(self.num_mc)]
            embeddings_out_zs = torch.stack(embeddings_out_zs)
            embeddings_out_zs = torch.transpose(embeddings_out_zs, 0, 1)
            # shape: batchsize, num_mc, c, w, h
            ndm_zs = embeddings_out_zs

            # compute the Bayesian coral loss
            ndm_loss = self.bayesian_coral.update(ndm_zs, ndm_zt)
        else:
            ndm_loss = Variable(torch.tensor(0)).cuda()

        # L_SL ,refers to Eq. (15)
        loss_xs_reconstruction = (loss_xs_denoising + scaled_kl)
        
        total_loss = self.source_weight * loss_xs_reconstruction  + self.bayesian_coral_weight * ndm_loss +  self.style_lambda * loss_style + self.xt_reconstruction_weight * xt_reconstruction_loss
        total_loss = total_loss.to(self.device)

        del ys_hat


        return loss_style, ndm_loss, loss_xs_reconstruction, xt_reconstruction_loss, scaled_kl, total_loss

    def discrinator_pass(self,xt,xs,y):
        '''
                In this version, we aim to improve the manner of bayesian coral.
                :param xt:
                :param y:
                :param batch_size:
                :return:
                '''
        # encoder features of generator 
        z_t= self.encoder_denoising(xt)
        embeddings_lists_zt = []
        
        output_zt_list = []
        # one pass
        for mc_run in range(1):
            input_zt = z_t
            # sample a instance from BNN layer 
            embedding_zt = self.mu(input_zt)
            embeddings_lists_zt.append(embedding_zt)
            output_zt = self.decoder_denoising_part1(embedding_zt)
            output_zt = self.regression(output_zt, return_kl=False)
            output_zt = F.relu(output_zt)
            output_zt_list.append(output_zt)

        # input the mlv
        yt_hat = torch.mean(torch.stack(output_zt_list), dim=0)
        ys_hat = y

        # no mlv
        if self.direct:
            ys_hat_mlv = ys_hat
            yt_hat_mlv = yt_hat
        else:
            yt_hat_mlv = mlv(yt_hat)
            ys_hat_mlv = mlv(ys_hat)

        d_fake = self.discriminator(yt_hat_mlv)
        d_real = self.discriminator(ys_hat_mlv)
        if self.lsgan:
            loss_D_real = self.criterionGAN(d_real, True)
            # Fake
            loss_D_fake = self.criterionGAN(d_fake, False)
            # Combined loss
            d_loss = (loss_D_real + loss_D_fake) * 0.5
        else:
            d_loss = -torch.mean(d_real) + torch.mean(d_fake)

        if self.gp_bool == True and self.lsgan == False:
            gp_loss = self.gp(ys_hat_mlv, yt_hat_mlv)
            loss_style = d_loss + gp_loss
        else:
            # use lsgan type, gp is set to 0
            gp_loss = 0
            loss_style = d_loss

        loss_style = loss_style.to(self.device)
        del ys_hat_mlv,yt_hat_mlv,yt_hat
        return loss_style,gp_loss


    def predict(self,x):
        z = self.encoder_denoising(x)

        embeddings_lists_zs = []
        output_zs_list = []
        for mc_run in range(20):
            input_zs = z

            embedding_zs = self.mu(input_zs)


            embeddings_lists_zs.append(embedding_zs)


            output_zs = self.decoder_denoising_part1(embedding_zs)
            output_zs = self.regression(output_zs,return_kl=False)
            output_zs = F.relu(output_zs)
            output_zs_list.append(output_zs)

        # mean
        output = torch.mean(torch.stack(output_zs_list), dim=0)
        if self.xt_reconstruction:
            xt_output = self.decoder_content(output)
        return output, xt_output
    def test_coral(self,xs_test,xt):

        # maintain the z is invariant by encoder
        z_t = self.encoder_denoising(xt)
        #

        z_st = self.encoder_denoising(xs_test)


        # print('-'*10,'pass')
        # bayesian layers

        embeddings_lists_zs_t = []
        embeddings_lists_zt = []
        kl_lists = []
        # print('-' * 10, 'bayesian test')
        for mc_run in range(30):

            input_zt = z_t
            input_zst = z_st

            embedding_zt = self.mu(input_zt)
            embedding_zst = self.mu(input_zst)

            #embeddings_lists_zs.append(embedding_zs)
            embeddings_lists_zt.append(embedding_zt)
            embeddings_lists_zs_t.append(embedding_zst)


        embeddings_lists_zt = torch.stack(embeddings_lists_zt)
        # metric learning
        embeddings_out_zt = []
        # print(embeddings_lists_zt.shape)
        for index in range(self.num_mc):
            # print(index)
            embeddings_out_zt.append(self.squeeze(embeddings_lists_zt[index]))
        # exit()
        embeddings_out_zt = torch.stack(embeddings_out_zt)
        #print(embeddings_out_zt.shape)
        embeddings_out_zt = torch.transpose(embeddings_out_zt, 0, 1)
        # shape: batchsize, num_mc, c, w, h
        ndm_zt = embeddings_out_zt


        embeddings_lists_zst = torch.stack(embeddings_lists_zs_t)
        # metric learning
        embeddings_out_zst = []

        for index in range(self.num_mc):
            embeddings_out_zst.append(self.squeeze(embeddings_lists_zst[index]))
        # exit()
        embeddings_out_zst = torch.stack(embeddings_out_zst)
        #print(embeddings_out_zst.shape)
        embeddings_out_zst = torch.transpose(embeddings_out_zst, 0, 1)
        # shape: batchsize, num_mc, c, w, h
        ndm_zst = embeddings_out_zst
        #exit()
        ndm_loss_indomain = self.bayesian_coral.compute(ndm_zst)
        ndm_loss_outdomain = self.bayesian_coral.compute(ndm_zt)

        return ndm_loss_outdomain,ndm_loss_indomain



"""
Whole Network
"""
##******************************************************************************************************************************
class WGAN_SACNN_AE(nn.Module):
    def __init__(self, N, root_path, version="v2"):
        super(WGAN_SACNN_AE, self).__init__()
        if torch.cuda.is_available():
            self.generator = SACNN(N, version).cuda()
            #self.discriminator = DISC().cuda()
            self.p_criterion = nn.MSELoss().cuda()
        else:
            self.generator = SACNN(N, version)
            self.discriminator = DISC()
            self.p_criterion = nn.MSELoss()
        #ae_path = root_path + "/AE/Ae_E279_val_Best.pkl"             ## The network has been trained to compute perceputal loss
        #Ae = AE()
        #self.ae = updata_ae(Ae, ae_path)

    def feature_extractor(self, image, model):
        model.eval()
        pred,y = model(image)
        return y

    def d_loss(self, x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        """
        generator loss
        """
        fake = self.generator(x)
        #d_fake = self.discriminator(fake)
        #g_loss = -torch.mean(d_fake)
        #mse_loss = self.p_criterion(x, y)
        #g_loss += mse_loss*100
        g_loss = 0
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        """
        percetual loss
        """
        fake = self.generator(x)
        real = y
        fake_feature = self.feature_extractor(fake, self.ae)
        real_feature = self.feature_extractor(real, self.ae)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

