'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-28 20:50:56
@LastEditors: GuoYi
'''
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import itertools
import time
import math
import pickle
import numpy as np
import scipy.io as sio
from measure import *
from numpy import *
import piq
from vif_utils import  vif,vif_spatial
from scipy.io import loadmat 
import torch
from torch import optim
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torchvision import models
class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        vgg = models.vgg19(pretrained=False) # .cuda()
        vgg.load_state_dict(torch.load('/media/ubuntu/88692f9c-d324-4895-8764-1cf202f9e6ac/chenkecheng/vgg19_new.pth'))
        #exit()
        vgg_pretrained_features = vgg.features
        '''
        keys_lists = list(vgg.state_dict().keys())
        #print(keys_lists)
        #exit()
        weights = np.load('/media/ubuntu/88692f9c-d324-4895-8764-1cf202f9e6ac/chenkecheng/vgg19.npy', allow_pickle=True,
                          encoding='latin1').item()
        layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                  'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                  'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                  'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        # print(self.vgg_params)
        for i, k in enumerate(layers):
            # print(i, k, weights[k][0].shape, weights[k][1].shape)
            # self.sess.run(self.vgg_params[ 2 *i].assign(weights[k][0]))
            vgg.state_dict()[keys_lists[2*i]] = weights[k][0]
            vgg.state_dict()[keys_lists[2*i+1]] = weights[k][1]
            #print(k)
            #print(weights[k][1])
        '''
        vgg.eval()
        self.requires_grad = requires_grad
        #self.slice1 = torch.nn.Sequential()
        #self.slice2 = torch.nn.Sequential()
        #self.slice3 = torch.nn.Sequential()
        #self.slice4 = torch.nn.Sequential()
        #self.slice5 = torch.nn.Sequential()

        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36])
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
    def __init__(self):

        super(VGGLoss, self).__init__()
        self.vgg = Vgg19_out(requires_grad=False)
        #self.criterion = nn.MSELoss(reduction='sum')
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]


    def forward(self, x, y,batch_size):
        #print(x.shape)
        #exit()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        #print(y_vgg)
        #exit()
        #print(x_vgg.shape)
        #exit()
        #for iter, (x_fea, y_fea) in enumerate(zip(x_vgg, y_vgg)):
            #print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
        loss = torch.sum(torch.square(x_vgg - y_vgg))/(batch_size*4*4*512)
        print(loss)
        exit()
        return  loss
## train function
##***********************************************************************************************************
def train(model, epoch, phase, optimizer_g, optimizer_d,dataloaders, args):
     loss_style_total = 0
     ndm_loss_total = 0
     total_loss_total = 0
     loss_xt_reconstruction_total = 0
     d_loss_all = 0
     kl_loss_all = 0

     total_iters = 0
     batch_size = args.batch_size[phase]
     for i, batch in enumerate(dataloaders):
          total_iters += 1

          time_batch_start = time.time()
          if args.baseline == True and args.baseline_type == 'n2n':
               full_image = batch["quarter_image_target"]
               quarter_image = batch["quarter_image_target_noise"]
               #quarter_image_target = batch["quarter_image_target"]
               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               #quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               #quarter_image_target = Variable(quarter_image_target).cuda()
          elif args.baseline == True and args.baseline_type == 'disc':
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               # quarter_image_target = batch["quarter_image_target"]
               full_image = torch.reshape(full_image, [-1, 1, 6, 6]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               # quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
          elif args.baseline == True and args.baseline_type == 'uda':
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               full_image_T = batch["full_image_T"]
               quarter_image_T = batch["quarter_image_T"]
               quarter_image_T_noise = batch["quarter_image_T_noise"]
               # quarter_image_target = batch["quarter_image_target"]
               full_image = torch.reshape(full_image, [-1, 1, 6, 6]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               full_image_T = torch.reshape(full_image_T, [-1, 1, 6, 6]).float()
               quarter_image_T = torch.reshape(quarter_image_T, [-1, 1, args.patch_size, args.patch_size]).float()
               quarter_image_T_noise = torch.reshape(quarter_image_T_noise, [-1, 1, args.patch_size, args.patch_size]).float()

               # quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               full_image_T = Variable(full_image_T).cuda()
               quarter_image_T = Variable(quarter_image_T).cuda()
               quarter_image_T_noise = Variable(quarter_image_T_noise).cuda()
          else:
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               quarter_image_target = batch["quarter_image_target"]

               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               quarter_image_target = Variable(quarter_image_target).cuda()
               if args.unpaired:
                    quarter_image_target_GT = batch["quarter_image_target_GT"]
                    quarter_image_target_GT = torch.reshape(quarter_image_target_GT,
                                                         [-1, 1, args.patch_size, args.patch_size])
                    quarter_image_target_GT = Variable(quarter_image_target_GT).cuda()

          '''
          else:
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               quarter_image_target = batch["quarter_image_target"]
               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               quarter_image_target = Variable(quarter_image_target).cuda()
          '''
          ##**********************************
          # (1)discriminator
          ##**********************************.
          if args.style:
               for p in model.discriminator.parameters():
                    p.requires_grad_(True)
               for p in model.generator.parameters():
                    p.requires_grad_(False)

               optimizer_d.zero_grad()
               model.discriminator.zero_grad()

               for _ in range(args.n_d_train):
                    if args.baseline:
                         if args.baseline_type == 'WGAN':
                              loss1, gp_loss = model.discrinator_pass(quarter_image,full_image)
                         elif args.baseline_type == 'RFM':
                              loss1, gp_loss = model.discrinator_pass(quarter_image_target, quarter_image, full_image)
                         else:
                              print('The name of the model is not found')
                              exit()
                    else:
                         if args.unpaired:
                              loss1, gp_loss = model.discrinator_pass(quarter_image_target, quarter_image, quarter_image_target_GT)
                         else:
                              loss1, gp_loss = model.discrinator_pass(quarter_image_target, quarter_image, full_image)


                    loss1.backward()
                    #for name, parms in model.named_parameters():
                    #     if name == "discriminator.model.6.bias":
                    #          print('-->name:', name)
                    #          print('-->para:', parms)
                    #          #print('-->grad_requirs:', parms.requires_grad)
                    #          #print('-->grad_value:', parms.grad)
                    #          print("===")
                    optimizer_d.step()

                    d_loss_all += (
                                 (loss1.item() - gp_loss) * quarter_image.size(0) / args.n_d_train)  ## loss = d_loss + gp_loss
                    del loss1


          for p in model.discriminator.parameters():
               p.requires_grad_(False)
          for p in model.generator.parameters():
               p.requires_grad_(True)

          optimizer_g.zero_grad()
          model.generator.zero_grad()

          for _ in range(args.n_g_train):
               if not args.baseline:
                    loss_style, ndm_loss, loss_xs_reconstruction, loss_xt_reconstruction, scaled_kl, total_loss = model.updata_v2(
                         quarter_image_target, quarter_image, full_image, batch_size, epoch=epoch)
               else:
                    if args.baseline == True and args.baseline_type == 'uda':
                         loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction, scaled_kl, total_loss = model.updata(
                              quarter_image, full_image, quarter_image_T, full_image_T, quarter_image_T_noise, batch_size)
                    else:
                         loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction, scaled_kl, total_loss = model.updata(
                              quarter_image, full_image, batch_size)
               #optimizer_g.zero_grad()


               total_loss.backward()

               optimizer_g.step()
               #for name, parms in model.named_parameters():
               #     if name == "decoder_content.layer1.conv2d.0.bias":
               #          print('-->name:', name)
               #          print('-->para:', parms)
               #          print('-->grad_requirs:', parms.requires_grad)
               #          print('-->grad_value:', parms.grad)
               #          print("===")
               #exit()
               #print(model.generator.state_dict()["4.layer1.conv2d.0.bias"])
          total_loss_total += total_loss.item()
          del total_loss




          if total_iters % 50 == 0:
               print('-'*20,'iter:{}'.format(total_iters),'-'*20)
               if args.baseline == True and (args.baseline_type == 'n2n' or args.baseline_type == 'disc') :
                    print('-' * 20,
                          'total_loss:{:.5f}'.format(
                               total_loss),
                          '-' * 20)
               else:
                    print('-'*20,'loss_style:{:.5f}, ndm_loss:{}, loss_xs_reconstruction:{:.5f}, loss_xt_reconstruction:{:.5f},kl_loss:{:.5f}'.format(loss_style, ndm_loss, loss_xs_reconstruction, loss_xt_reconstruction,scaled_kl),'-'*20)

          if args.use_p_loss is False:
               p_loss = 0
          loss_style_total += loss_style.item()
          ndm_loss_total += ndm_loss.item()

          loss_xt_reconstruction_total += loss_xt_reconstruction.item()
          kl_loss_all += scaled_kl.item()


     print(total_iters)
     loss_d_loss = d_loss_all / total_iters
     loss_kl = kl_loss_all / total_iters
     loss_style_return = loss_style_total/total_iters
     ndm_loss_return = ndm_loss_total/total_iters
     total_loss_total_return = total_loss_total/total_iters
     loss_xt_reconstruction_return = loss_xt_reconstruction_total/total_iters

     return loss_d_loss,loss_kl,loss_style_return,ndm_loss_return,total_loss_total_return,loss_xt_reconstruction_return

##*********************************************************************************************************************
def train_cycle(model, epoch, phase, optimizer_g, optimizer_d_A,optimizer_d_B,dataloaders, args):
     loss_style_total = 0
     ndm_loss_total = 0
     loss_xt_reconstruction_total = 0
     loss_xs_reconstruction_total = 0
     d_loss_all = 0
     gp_loss_all = 0

     total_iters = 0
     batch_size = args.batch_size[phase]
     for i, batch in enumerate(dataloaders):
          total_iters += 1

          time_batch_start = time.time()
          #if args.baseline:
          full_image = batch["full_image"]
          quarter_image = batch["quarter_image"]
          quarter_image_target = batch["quarter_image_target"]
          full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size])
          quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
          quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
          full_image = Variable(full_image).cuda()
          quarter_image = Variable(quarter_image).cuda()
          quarter_image_target = Variable(quarter_image_target).cuda()
          '''
          else:
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               quarter_image_target = batch["quarter_image_target"]
               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               quarter_image_target = Variable(quarter_image_target).cuda()
          '''
          ##**********************************
          # (1)discriminator
          ##**********************************

          optimizer_g.zero_grad()
          model.netG_A.zero_grad()
          model.netG_B.zero_grad()

          loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction, scaled_kl, total_loss = model.updata(
                    full_image, quarter_image_target,batch_size)
          optimizer_g.zero_grad()
          total_loss.backward()
          optimizer_g.step()
          # dis A
          optimizer_d_A.zero_grad()
          model.netD_A.zero_grad()

          loss1 = model.dis_A(full_image)
          loss1.backward()
          optimizer_d_A.step()
          d_loss_all += (
                  (loss1 - 0) * quarter_image.size(0) / args.n_d_train)  ## loss = d_loss + gp_loss
          gp_loss_all += (0 * quarter_image.size(0) / args.n_d_train)

          optimizer_d_B.zero_grad()
          model.netD_B.zero_grad()
          loss2 = model.dis_B(quarter_image_target)
          loss2.backward()
          optimizer_d_B.step()

          #if i == 10:
          #     exit()
          if total_iters % 50 == 0:
               print('-'*20,'iter:{}'.format(total_iters),'-'*20)
               print('-'*20,'loss_total:{:.5f},loss_dis_A:{:.5f},loss_dis_B:{:.5f} '.format(total_loss,loss1,loss2),'-'*20)

          if args.use_p_loss is False:
               p_loss = 0
          loss_style_total += loss_style
          ndm_loss_total += ndm_loss
          loss_xt_reconstruction_total += total_loss
          loss_xs_reconstruction_total += loss_xs_reconstruction


          #if i>0 and math.fmod(i, args.show_batch_num[phase]) == 0:
          #     print("Epoch {} Batch {}-{} {}, Time:{:.4f}s".format(epoch+1,
          #     i-args.show_batch_num[phase], i, phase, (time.time()-time_batch_start)*args.show_batch_num[phase]))
     print(total_iters)
     loss_d_loss = d_loss_all / total_iters
     loss_gp = gp_loss_all / total_iters
     loss_style_return = loss_style_total/total_iters
     ndm_loss_return = ndm_loss_total/total_iters
     loss_xt_reconstruction_return = loss_xt_reconstruction_total/total_iters
     loss_xs_reconstruction_return = loss_xs_reconstruction_total/total_iters

     return loss_d_loss,loss_gp,loss_style_return,ndm_loss_return,loss_xt_reconstruction_return,loss_xs_reconstruction_return

## Train
##***********************************************************************************************************
def train_model(model,
                dataloaders,
                args): 
     psnr = []
     ssim = []
     vif = []
     vis = []
     if args.baseline == True and args.baseline_type == 'clycle':
          optimizer_g = optim.Adam(itertools.chain(model.netG_A.parameters(), model.netG_B.parameters()), lr=args.lr,betas=(0.9, 0.999))
     elif args.baseline == True and args.baseline_type == 'disc':
          optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
     else:
          optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr,betas=(0.9, 0.999))
     StepLR = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.9)
     #print(type(model.generator.state_dict()))

     #for param_tensor in model.generator.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值

     #exit()

     if args.style:
          optimizer_d = optim.Adam(model.discriminator.parameters(), lr=3*args.lr, betas=(0.9, 0.999))
     elif args.baseline == True and args.baseline_type == 'clycle':
          optimizer_d_A = optim.Adam(model.netD_A.parameters(), lr= args.lr, betas=(0.9, 0.999))
          optimizer_d_B = optim.Adam(model.netD_B.parameters(), lr= args.lr, betas=(0.9, 0.999))
     else:
          optimizer_d = None
     # optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr, alpha=0.9)
     #optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr, alpha=0.9)

     if args.re_load is False:
          print("\nInit Start**")
          #model.apply(weights_init)
          print("******Init End******\n")
          if args.baseline == True and args.baseline_type == 'uda':
               if args.region == 'head':
                    pretrained_disc = torch.load(
                         '/root/autodl-tmp/results_MAE_Test_baseline/training_on_head_validate_on_abd_new_region/search/Baseline-disc-2022-12-24 14:48:49.684653seed_0_mc_9999_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/WGAN_SACNN_AE_Ebaseline.pkl')
                    model_dict_disc = model.discriminator.state_dict()
                    pretrained_dict = {k.replace('generator.',''): v for k, v in pretrained_disc.items()}
                    model_dict_disc.update(pretrained_dict)
                    model.discriminator.load_state_dict(model_dict_disc)
                    print('-' * 10, '...loading disc sucecuss...', '-' * 10)

                    pretrained_generator = torch.load(
                         '/root/autodl-tmp/results_MAE_Test_baseline/training_on_head_validate_on_abd_new_region/search/Baseline-plain-2022-12-24 11:21:07.474314seed_0_mc_9999_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/WGAN_SACNN_AE_Ebaseline.pkl')
                    model_dict_generator = model.generator.state_dict()
                    pretrained_dict = {k.replace('generator.',''): v for k, v in pretrained_generator.items() if k in model_dict_generator}
                    model_dict_generator.update(pretrained_dict)
                    model.generator.load_state_dict(model_dict_generator)
                    print('-'*10,'...loading generator sucecuss...','-'*10)


                    for name, parameter in model.discriminator.named_parameters():
                         parameter.requires_grad = False
               else:
                    pretrained_disc = torch.load(
                         '/root/autodl-tmp/results_MAE_final/training_on_abd_validate_on_head_new_region/search/Baseline-disc-2022-12-27 23:35:06.018708seed_0_mc_9999_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/Baseline-disc--100.pkl')
                    model_dict_disc = model.discriminator.state_dict()
                    pretrained_dict = {k.replace('generator.', ''): v for k, v in pretrained_disc.items()}
                    model_dict_disc.update(pretrained_dict)
                    model.discriminator.load_state_dict(model_dict_disc)
                    print('-' * 10, '...loading disc sucecuss...', '-' * 10)

                    pretrained_generator = torch.load(
                         '/root/autodl-tmp/results_MAE_final/training_on_abd_validate_on_head_new_region/search/Baseline-baseline-2022-12-27 21:37:37.951117seed_0_mc_9999_xtRe_0_Coral_0_mope_delta_0.1_posterior_rho_-4/model/Baseline-baseline--100.pkl')
                    model_dict_generator = model.generator.state_dict()
                    pretrained_dict = {k.replace('generator.', ''): v for k, v in pretrained_generator.items() if
                                       k in model_dict_generator}
                    model_dict_generator.update(pretrained_dict)
                    model.generator.load_state_dict(model_dict_generator)
                    print('-' * 10, '...loading generator sucecuss...', '-' * 10)

                    for name, parameter in model.discriminator.named_parameters():
                         parameter.requires_grad = False
     else:
          print("Re_load is True !")
          model, optimizer_g, optimizer_d = updata_model(model, optimizer_g, optimizer_d, args)
          
     losses = {x: torch.zeros(args.epoch_num, 8) for x in ["train", "val"]}
     # if args.re_load is True:
     #      losses = {x: torch.from_numpy(loadmat(args.loss_path + "/losses.mat")[x]) for x in ["train", "val"]}
     #      print("Load losses Done")

     ##********************************************************************************************************************
     time_all_start = time.time()
     # for epoch in range(args.old_index if args.re_load else 0, args.epoch_num):
     for epoch in tqdm(range(args.epoch_num)):
          time_epoch_start = time.time()
          print("-" * 60)
          print(".........Training and Val epoch {}, all {} epochs..........".format(epoch+1, args.epoch_num))
          print("-" * 60)


          ##//////////////////////////////////////////////////////////////////////////////////////////////
          # for phase in ["train", "val"]:
          ## if model.train, BN and Dropout operations are used. Instead, they are freezed
          for phase in ["train"]:
               print("\n=========== Now, Start {}===========".format(phase))
               if phase is "train":
                    model.train()
               elif phase is "val":
                    model.eval()
               if args.baseline == True and args.baseline_type == 'clycle':
                    loss_d, loss_gp, loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction = train_cycle(model,
                                                                                                                  epoch,
                                                                                                                  phase,
                                                                                                                  optimizer_g,
                                                                                                                  optimizer_d_A,
                                                                                                                  optimizer_d_B,
                                                                                                                  dataloaders[
                                                                                                                       phase],
                                                                                                                  args)
               else:
                    loss_d,loss_kl,loss_style, ndm_loss, loss_total, loss_xt_reconstruction = train(model, epoch, phase, optimizer_g,  optimizer_d,dataloaders[phase], args)
               losses[phase][epoch] = torch.tensor([loss_d,loss_kl,loss_style, ndm_loss, loss_total, loss_xt_reconstruction,0,0])
          StepLR.step()

          ##//////////////////////////////////////////////////////////////////////////////////////////////
          label = 0

          if  epoch != 0 and epoch % args.validate_epoch ==0 and args.validate == True: # epoch != 0 and epoch % args.validate_epoch ==0 and
               model.eval()
               data_eval = dataloaders['val']
               full_image_set = []
               low_image_set = []
               predict_image_set = []
               #for i, batch in enumerate(data_eval):
               batch = data_eval
               time_batch_start = time.time()
               full_image = batch["full_image"]

               #exit()
               full_image_set.append(full_image)
               quarter_image = batch["quarter_image"]
               total_test = quarter_image.shape[0]
               low_image_set.append(quarter_image)
               for index in tqdm(range(total_test)):

                    with torch.no_grad():
                         quarter_image_test = torch.tensor(quarter_image[index])

                         quarter_image_test = torch.reshape(quarter_image_test, [1, 1, 512, 512])
                         if args.use_cuda:
                              #full_image = Variable(full_image).cuda()
                              quarter_image_test = Variable(quarter_image_test).cuda()
                         else:
                              #full_image = Variable(full_image)
                              pass
                         quarter_image_test = Variable(quarter_image_test)

                         predicts = model.predict(quarter_image_test)
                         predict_image_set.append(predicts.cpu().detach().numpy())

               full_image_set = np.concatenate(full_image_set,axis=0)
               #print(full_image_set.shape)
               full_image_set = full_image_set.reshape([-1,512,512,1])
               #exit()
               #.reshape([-1,512,512,1])
               predict_image_set = np.concatenate(predict_image_set,axis=0).reshape([-1,512,512,1])
               low_image_set = np.concatenate(low_image_set,axis=0).reshape([-1,512,512,1])

               #np.save(args.show_image + '/region_{}_p_{}_m_{}_{}.npy'.format(args.target,args.p_lambda,args.m_gamma,epoch),predict_image_set)

               #if epoch == args.validate_epoch or epoch == (args.validate_epoch+1):
               #     np.save(args.show_image + '/low.npy'.format(epoch), low_image_set)
               #     np.save(args.show_image + '/full.npy'.format(epoch), full_image_set)

               visualize(args,epoch,low_image_set,full_image_set,predict_image_set)
               torch.cuda.empty_cache()
               model.train()

               torch.save(model.generator.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))
               #torch.save(optimizer_g.state_dict(), args.optimizer_path + "/" + args.optimizer_g_name + "{}.pkl".format(epoch))
               #torch.save(optimizer_d.state_dict(), args.optimizer_path + "/" + args.optimizer_d_name + "{}.pkl".format(epoch))

          if epoch != 0 and args.validate_coral == True:
               #data_eval = dataloaders['train']
               xs = args.test_patch_source
               xt = args.test_patch_target
               ndm_total_indomain = 0
               ndm_total_outdomain = 0
               model.eval()
               num = 0
               with torch.no_grad():

                         num = 1
                         #quarter_image = batch["quarter_image"]

                         #quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
                         xs_image = torch.tensor(xs)
                         xt_image = torch.tensor(xt)

                         if args.use_cuda:
                              xt_image = Variable(xt_image).cuda()
                              xs_image = Variable(xs_image).cuda()


                         loss_outdomain,loss_indomain = model.test_coral(xs_image,xt_image)
                         ndm_total_indomain += loss_indomain
                         ndm_total_outdomain += loss_outdomain


               losses['val'][epoch] = torch.tensor(
                    [0, 0,0,0,0,0,ndm_total_indomain,ndm_total_outdomain])
               model.train()





          if  epoch != 0 and epoch % args.validate_epoch ==0 and epoch >= args.validate_shreshold and args.test == True: #math.fmod(epoch, args.validate_epoch) == 0 and epoch != 0:
               model.eval()
               data_eval = dataloaders['val']
               full_image_set = []
               low_image_set = []
               predict_image_set = []
               #for i, batch in enumerate(data_eval):
               #exit()
               #full_image_set.append(full_image)



               for _,batch in enumerate(data_eval):

                    with torch.no_grad():
                         full_image = batch["full_image"]
                         full_image_set.append(full_image)

                         quarter_image = batch["quarter_image"]
                         low_image_set.append(quarter_image)


                         quarter_image = torch.reshape(quarter_image, [-1, 1, 512, 512]).float()
                         quarter_image = Variable(quarter_image).cuda()



                         if args.use_cuda:
                              #full_image = Variable(full_image).cuda()
                              quarter_image = Variable(quarter_image).cuda()
                         else:
                              #full_image = Variable(full_image)
                              pass

                         predicts = model.predict(quarter_image)
                         predict_image_set.append(predicts.cpu().detach().numpy())

               full_image_set = np.concatenate(full_image_set,axis=0)
               #print(full_image_set.shape)
               full_image_set = full_image_set.reshape([-1,1,512,512])

               predict_image_set = np.concatenate(predict_image_set,axis=0).reshape([-1,1,512,512])
               low_image_set = np.concatenate(low_image_set,axis=0).reshape([-1,512,512,1])

               np.save(args.show_image + '/region_{}_p_{}_m_{}_{}.npy'.format(args.region, args.p_lambda, args.m_gamma,
                                                                              epoch), predict_image_set)
               if epoch == 80 or epoch == 81:
                    np.save(args.show_image + '/low.npy'.format(epoch), low_image_set)
                    np.save(args.show_image + '/full.npy'.format(epoch), full_image_set)
               mean_psnr, mean_ssim, mean_vif, mean_vis = visualize(args,epoch,low_image_set,full_image_set,predict_image_set)
               psnr.append(mean_psnr)
               ssim.append(mean_ssim)
               vif.append(mean_vif)
               vis.append(mean_vis)
               torch.cuda.empty_cache()
               model.train()
               if args.baseline == True and args.baseline_type == 'clycle':
                    torch.save(model.netG_A.state_dict(), args.model_path + "/" + args.model_name + "-{}.pkl".format(epoch))
               else:
                    torch.save(model.generator.state_dict(),
                               args.model_path + "/" + args.model_name + "-{}.pkl".format(epoch))
               #torch.save(optimizer_g.state_dict(), args.optimizer_path + "/" + args.optimizer_g_name + "{}.pkl".format(epoch))
               #torch.save(optimizer_d.state_dict(), args.optimizer_path + "/" + args.optimizer_d_name + "{}.pkl".format(epoch))
          data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in losses.items()}
          sio.savemat(args.loss_path + "/{}_losses.mat".format(args.name), mdict = data_save)
          print('-' * 20, 'D Loss:{:.6f}'.format(losses[phase][epoch][0].item()),'MSE Loss:{:.6f}'.format(losses[phase][epoch][5].item()),'-'*20)
          print("Time for epoch {} : {:.4f}min".format(epoch+1, (time.time()-time_epoch_start)/60))
          print("Time for ALL : {:.4f}h\n".format((time.time()-time_all_start)/3600))

     ##********************************************************************************************************************

     print("\nTrain Completed!! Time for ALL : {:.4f}h".format((time.time()-time_all_start)/3600))
     number = len(psnr)
     return sum(psnr)/number, sum(ssim)/number, sum(vif) / number, sum(vis)/number

## Init the model
##***********************************************************************************************************
def weights_init(m):
     classname = m.__class__.__name__
     if classname.find("Conv2d") != -1:
          init.xavier_normal_(m.weight.data)
          if m.bias is not None:
               init.constant_(m.bias.data, 0)
          print("Init {} Parameters.................".format(classname))
     if classname.find("ConvTranspose2d") != -1:
          init.xavier_normal_(m.weight.data)
          if m.bias is not None:
               init.constant_(m.bias.data, 0)
          print("Init {} Parameters.................".format(classname))
     if classname.find("Linear") != -1:
          init.xavier_normal(m.weight)
          print("Init {} Parameters.................".format(classname))
     else:
          print("{} Parameters Do Not Need Init !!".format(classname))
def measure(args,low,full,predict):
     psnr = []
     ssim = []
     GMSD = []
     DSS = []
     total_imges = low.shape[0]

     if args.target == 'ISICDM':

          top1 = [295,220,200]
          top2 = [415,400,400]
          bottom1 = [180,100,300]
          bottom2 = [350,450,450]
          predict = np.clip(predict, 0, 1)
          for i in range(low.shape[0]):

               denoising_test = renormalization(predict[i, top1[i]:top2[i], bottom1[i]:bottom2[i]],-1000,400)
               full_test = renormalization(full[i, top1[i]:top2[i], bottom1[i]:bottom2[i]],-1000,400)
               psnr.append(np.mean(denoising_test))
               ssim.append(np.mean(full_test))
               GMSD.append(np.std(denoising_test))
               DSS.append(np.std(full_test))
          return mean(psnr), mean(ssim), mean(GMSD), mean(DSS)


     if args.validate:
          predict = np.clip(predict, 0, 1)
          denoising_test = torch.tensor(predict[:,50:450, 50:450]).cuda()
          full_test = torch.tensor(full[:, 50:450, 50:450]).cuda()
          for i in tqdm(range(total_imges)):
               psnr.append(piq.psnr(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
               ssim.append(piq.ssim(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
               GMSD.append(piq.GMSDLoss(data_range=1.)(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
               DSS.append(piq.DSSLoss(data_range=1.)(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())

          return mean(psnr), mean(ssim), mean(GMSD), mean(DSS)

     else:
          predict = np.clip(predict,0,1)
          if args.validate_region == 'head':
               denoising_test = torch.tensor(predict[:, :, 100:400, 100:400]).cuda()
               full_test = torch.tensor(full[:, :, 100:400, 100:400]).cuda()
          else:
               denoising_test = torch.tensor(predict[:,:,50:400,50:450]).cuda()
               full_test = torch.tensor(full[:,:,50:400,50:450]).cuda()
          psnr_mean =piq.psnr(denoising_test, full_test,reduction='mean').data.cpu()
          ssim_mean = piq.ssim(denoising_test, full_test,reduction='mean').data.cpu()
          for i in tqdm(range(total_imges)):
               if args.validate_region == 'head':
                    vif_.append(piq.vif_p(denoising_test[i].resize(1, 1, 300, 300), full_test[i].resize(1, 1, 300, 300),
                                          reduction='mean').data.cpu())
                    vis.append(piq.VSILoss(data_range=1.)(denoising_test[i].resize(1, 1, 300, 300),
                                                          full_test[i].resize(1, 1, 300, 300)).data.cpu())
               else:
                    vif_.append(piq.vif_p(denoising_test[i].resize(1,1,350,400), full_test[i].resize(1,1,350,400),reduction='mean').data.cpu())
                    vis.append(piq.VSILoss(data_range=1.)(denoising_test[i].resize(1,1,350,400), full_test[i].resize(1,1,350,400)).data.cpu())
          return psnr_mean, ssim_mean, mean(GMSD), mean(vis)

     return mean(psnr),mean(ssim),mean(vif_),mean(DSS)

def renormalization(data,min_hu,max_hu):

    output = data*(max_hu - min_hu) + min_hu
    return output

def windowing(image,min_bound=-1000,max_bound=1000):
    output = (image-min_bound)/(max_bound-min_bound)
    output[output<0]=0
    output[output>1]=1
    return output

def visualize(args,epoch,low,full,predict):

     mean_psnr, mean_ssim,mean_vif,mean_vis = measure(args,low,full, predict)
     if args.validate:
          fig_titles = ['LDCT', 'Denoised by Ours', 'NDCT']
          plt.figure()
          f, axs = plt.subplots(figsize=(40, 40), nrows=1, ncols=3)
          if args.target == 'ISICDM':
               for num in range(0, 3):
                    ldct = np.squeeze(low[num])
                    ndct = np.squeeze(full[num])
                    # plt.imshow(ldct,cmap=plt.cm.gray)
                    # plt.show()
                    denoising = np.squeeze(predict[num])
                    denoising = np.clip(denoising, 0, 1.)

                    # psnr_before_total += psnr_scaled(ldct,ndct)
                    # psnr_d_total += psnr_scaled(denoising,ndct)
                    # ssim_before_total += calculate_ssim(ldct*255,ndct*255)
                    # ssim_d_total += calculate_ssim(denoising*255,ndct*255)
                    # vif_before_total += vif_spatial(ldct[100:300],ndct[100:300])
                    # vif_d_total += vif_spatial(denoising[100:300],ndct[100:300])

                    for s in range(1):
                         ldct = windowing(renormalization(ldct, -1000, 400), -160, 240)
                         axs[0].imshow(ldct, cmap=plt.cm.gray)
                         # print('ckc')
                         # axs[0].set_title(fig_titles[0], fontsize=30)
                         axs[0].axis('off')

                         denoising = windowing(renormalization(denoising, -1000, 400), -160, 240)
                         axs[1].imshow(denoising, cmap=plt.cm.gray)
                         # axs[1].set_title(fig_titles[1], fontsize=30)
                         axs[1].axis('off')

                         # imshow
                         ndct = windowing(renormalization(ndct, -1000, 400), -160, 240)
                         axs[2].imshow(ndct, cmap=plt.cm.gray)
                         # axs[2].set_title(fig_titles[2], fontsize=30)
                         axs[2].axis('off')

                         plt.savefig(args.show_image + '/test_result_{}_{}.png'.format(num, epoch),
                                     bbox_inches='tight', pad_inches=0.0)
                         # plt.show()
          else:
               for num in range(206,209):
                    ldct = np.squeeze(low[num])
                    ndct = np.squeeze(full[num])
                    #plt.imshow(ldct,cmap=plt.cm.gray)
                    #plt.show()
                    denoising = np.squeeze(predict[num])
                    denoising = np.clip(denoising,0,1.)


                    #psnr_before_total += psnr_scaled(ldct,ndct)
                    #psnr_d_total += psnr_scaled(denoising,ndct)
                    #ssim_before_total += calculate_ssim(ldct*255,ndct*255)
                    #ssim_d_total += calculate_ssim(denoising*255,ndct*255)
                    #vif_before_total += vif_spatial(ldct[100:300],ndct[100:300])
                    #vif_d_total += vif_spatial(denoising[100:300],ndct[100:300])



                    for s in range(1):
                         ldct = windowing(renormalization(ldct,-1000,400),-160,240)
                         axs[0].imshow(ldct, cmap=plt.cm.gray)
                         # print('ckc')
                         #axs[0].set_title(fig_titles[0], fontsize=30)
                         axs[0].axis('off')

                         denoising = windowing(renormalization(denoising, -1000, 400), -160, 240)
                         axs[1].imshow(denoising, cmap=plt.cm.gray)
                         #axs[1].set_title(fig_titles[1], fontsize=30)
                         axs[1].axis('off')

                         # imshow
                         ndct = windowing(renormalization(ndct, -1000, 400), -160, 240)
                         axs[2].imshow(ndct, cmap=plt.cm.gray)
                         #axs[2].set_title(fig_titles[2], fontsize=30)
                         axs[2].axis('off')

                         plt.savefig(args.show_image + '/test_result_{}_{}.png'.format(num,epoch),
                                     bbox_inches='tight', pad_inches=0.0)
                         #plt.show()

     log_file = open(args.measure_path + '/{}_measure.txt'.format(args.name), 'a')
     log_file.write('p_d:{:.4f},ssim_d:{:.5f},vif_d:{:.5f},vis_d:{:.5f}\n'.format(mean_psnr, mean_ssim,mean_vif,mean_vis))

     log_file.close()
     return mean_psnr, mean_ssim, mean_vif, mean_vis

## Updata old model
##***********************************************************************************************************
def updata_model(model, optimizer_g, optimizer_d, args):
     print("Please set the path of expected model!")
     time.sleep(3)
     model_reload_path = args.old_modle_path
     optimizer_g_reload_path = args.old_optimizer_path + "/" + args.old_optimizer_g_name + ".pkl"
     optimizer_d_reload_path = args.old_optimizer_path + "/" + args.old_optimizer_d_name + ".pkl"

     if os.path.isfile(model_reload_path):
          print("Loading previously trained network...")
          print("Load model:{}".format(model_reload_path))
          checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
          model_dict = model.generator.state_dict()
          checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
          model_dict.update(checkpoint)
          model.generator.load_state_dict(model_dict)
          del checkpoint
          torch.cuda.empty_cache()
          if args.use_cuda:
               model = model.cuda()
          print("Done Reload!")
     else:
          print("Can not reload model....\n")
          time.sleep(10)
          sys.exit(0)
     '''
     if os.path.isfile(optimizer_g_reload_path):
          print("Loading previous optimizer...")
          print("Load optimizer:{}".format(optimizer_g_reload_path))
          checkpoint = torch.load(optimizer_g_reload_path, map_location = lambda storage, loc: storage)
          optimizer_g.load_state_dict(checkpoint)
          del checkpoint
          torch.cuda.empty_cache()
          print("Done Reload!")
     else:
          print("Can not reload optimizer_g....\n")
          time.sleep(10)
          sys.exit(0)

     if os.path.isfile(optimizer_d_reload_path):
          print("Loading previous optimizer...")
          print("Load optimizer:{}".format(optimizer_d_reload_path))
          checkpoint = torch.load(optimizer_d_reload_path, map_location = lambda storage, loc: storage)
          optimizer_d.load_state_dict(checkpoint)
          del checkpoint
          torch.cuda.empty_cache()
          print("Done Reload!")
     else:
          print("Can not reload optimizer_d....\n")
          time.sleep(10)
          sys.exit(0)
     '''
     return model, optimizer_g, optimizer_d
