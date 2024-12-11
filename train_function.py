
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
from scipy.io import loadmat 
import torch
from torch import optim
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torchvision import models

from utils import updata_model,vif, renormalization,windowing, measure
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
          if args.baseline == True and args.baseline_type == 'n2n':
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
          elif args.baseline == True and args.baseline_type == 'disc':
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               full_image = torch.reshape(full_image, [-1, 1, 6, 6]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
          elif args.baseline == True and args.baseline_type == 'uda':
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               full_image_T = batch["full_image_T"]
               quarter_image_T = batch["quarter_image_T"]
               quarter_image_T_noise = batch["quarter_image_T_noise"]
               full_image = torch.reshape(full_image, [-1, 1, 6, 6]).float()
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size]).float()
               full_image_T = torch.reshape(full_image_T, [-1, 1, 6, 6]).float()
               quarter_image_T = torch.reshape(quarter_image_T, [-1, 1, args.patch_size, args.patch_size]).float()
               quarter_image_T_noise = torch.reshape(quarter_image_T_noise, [-1, 1, args.patch_size, args.patch_size]).float()
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               full_image_T = Variable(full_image_T).cuda()
               quarter_image_T = Variable(quarter_image_T).cuda()
               quarter_image_T_noise = Variable(quarter_image_T_noise).cuda()
          else:
               # our model training loop
               full_image = batch["full_image"]
               quarter_image = batch["quarter_image"]
               quarter_image_target = batch["quarter_image_target"]

               full_image = torch.reshape(full_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image = torch.reshape(quarter_image, [-1, 1, args.patch_size, args.patch_size])
               quarter_image_target = torch.reshape(quarter_image_target, [-1, 1, args.patch_size, args.patch_size])
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
               quarter_image_target = Variable(quarter_image_target).cuda()
               # unpaired means that we can obtain the NDCT images from target domain, but the NDCT is unparied with LDCT 
               if args.unpaired:
                    quarter_image_target_GT = batch["quarter_image_target_GT"]
                    quarter_image_target_GT = torch.reshape(quarter_image_target_GT,
                                                         [-1, 1, args.patch_size, args.patch_size])
                    quarter_image_target_GT = Variable(quarter_image_target_GT).cuda()
     
          ##**********************************
          # step #1 update the discriminator
          ##**********************************.
          if args.style:
               # activate discriminator and deactivate generator
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
                    optimizer_d.step()
                    d_loss_all += (
                                 (loss1.item() - gp_loss) * quarter_image.size(0) / args.n_d_train)  ## loss = d_loss + gp_loss
                    del loss1

          ##**********************************
          # step #2 update the generator
          ##**********************************.

          # activate generator and deactivate dis
          if args.style == True:
               for p in model.discriminator.parameters():
                    p.requires_grad_(False)
          for p in model.generator.parameters():
               p.requires_grad_(True)

          optimizer_g.zero_grad()
          model.generator.zero_grad()

          for _ in range(args.n_g_train):
               if not args.baseline:
                    # our training loop for generator
                    # loss_style: generator loss based on MLV, refers to the second term of Eq. (13)
                    # ndm_loss: Bayesian coral, i.e., bayesian uncertainty loss, refers to Eq. (6)-(8)
                    # loss_xt_reconstruction: self-recontruction loss, refers to Eq. (9)
                    # scaled_kl: KL divergence loss in BNN training, refers to last two terms of Eq. (15)
                    loss_style, ndm_loss, loss_xs_reconstruction, loss_xt_reconstruction, scaled_kl, total_loss = model.updata_v2(
                         quarter_image_target, quarter_image, full_image, batch_size, epoch=epoch)
               else:
                    if args.baseline == True and args.baseline_type == 'uda':
                         loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction, scaled_kl, total_loss = model.updata(
                              quarter_image, full_image, quarter_image_T, full_image_T, quarter_image_T_noise, batch_size)
                    else:
                         loss_style, ndm_loss, loss_xt_reconstruction, loss_xs_reconstruction, scaled_kl, total_loss = model.updata(
                              quarter_image, full_image, batch_size)



               total_loss.backward()
               optimizer_g.step()
          total_loss_total += total_loss.item()
          del total_loss


          if total_iters % 50 == 0:
               print('-'*20,'iter:{}'.format(total_iters),'-'*20)
               if args.baseline == True and (args.baseline_type == 'n2n' or args.baseline_type == 'disc') :
                    print('-' * 20,'total_loss:{:.5f}'.format(
                                    loss_xs_reconstruction),'-' * 20) if args.baseline_type == 'n2n' else print('-' * 20,
                               'total_loss:{:.5f}'.format(
                                    total_loss_total),
                               '-' * 20)
               else:
                    print('-'*20,'loss_style:{:.5f}, ndm_loss:{}, loss_xs_reconstruction:{:.5f}, loss_xt_reconstruction:{:.5f},kl_loss:{:.5f}'.format(loss_style, ndm_loss, loss_xs_reconstruction, loss_xt_reconstruction,scaled_kl),'-'*20)


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
     # define the optimizer for generator
     if args.baseline == True and args.baseline_type == 'clycle':
          optimizer_g = optim.Adam(itertools.chain(model.netG_A.parameters(), model.netG_B.parameters()), lr=args.lr,betas=(0.9, 0.999))
     elif args.baseline == True and args.baseline_type == 'disc':
          optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
     else:
          optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr,betas=(0.9, 0.999))
     StepLR = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.9)
     
     # define the optimizer for discriminator
     # style is true, means the MLV is used
     if args.style:
          optimizer_d = optim.Adam(model.discriminator.parameters(), lr=3*args.lr, betas=(0.9, 0.999))
     elif args.baseline == True and args.baseline_type == 'clycle':
          optimizer_d_A = optim.Adam(model.netD_A.parameters(), lr= args.lr, betas=(0.9, 0.999))
          optimizer_d_B = optim.Adam(model.netD_B.parameters(), lr= args.lr, betas=(0.9, 0.999))
     else:
          optimizer_d = None
     
     
     # reload model 
     if args.re_load is False:
          print("\nInit Start**")
          #model.apply(weights_init)
          print("******Init End******\n")
          if args.baseline == True and args.baseline_type == 'uda':

                    pretrained_disc = torch.load(
                         '/hdd/ckc/CT/DA-CT/results_MAE_final/training_on_AAPM_validate_on_ISICDM_new_region/unpaired=False/0726/disc/Baseline-disc-p=1-m=1-2023-07-26 22:05:38.489259xsRe_0.1_mc_9999_xtRe_5_Coral_5_mope_delta_0.1_posterior_rho_-4/model/Baseline-disc-p=1-m=1-100.pkl')
                    model_dict_disc = model.discriminator.state_dict()
                    pretrained_dict = {k.replace('generator.',''): v for k, v in pretrained_disc.items()}
                    model_dict_disc.update(pretrained_dict)
                    model.discriminator.load_state_dict(model_dict_disc)
                    print('-' * 10, '...loading disc sucecuss...', '-' * 10)

                    pretrained_generator = torch.load(
                         '/hdd/ckc/CT/DA-CT/results_MAE_final/training_on_AAPM_validate_on_AAPM_5_new_region/unpaired=False/0724/deter_CPCE/Baseline-None-p=1-m=1-2023-07-24 17:33:39.309093xsRe_1_mc_9999_xtRe_1_Coral_1_mope_delta_0.1_posterior_rho_-4/model/Baseline-None-p=1-m=1-100.pkl')
                    model_dict_generator = model.generator.state_dict()
                    pretrained_dict = {k.replace('generator.',''): v for k, v in pretrained_generator.items() if k in model_dict_generator}
                    model_dict_generator.update(pretrained_dict)
                    model.generator.load_state_dict(model_dict_generator)
                    print('-'*10,'...loading generator sucecuss...','-'*10)


                    for name, parameter in model.discriminator.named_parameters():
                         parameter.requires_grad = False

     else:
          print("Re_load is True !")
          model, optimizer_g, optimizer_d = updata_model(model, optimizer_g, optimizer_d, args)
          
     losses = {x: torch.zeros(args.epoch_num, 8) for x in ["train", "val"]}

     ##********************************************************************************************************************
     time_all_start = time.time()
     for epoch in tqdm(range(args.epoch_num)):
          time_epoch_start = time.time()
          print("-" * 60)
          print(".........Training and Val epoch {}, all {} epochs..........".format(epoch+1, args.epoch_num))
          print("-" * 60)
          
          for phase in ["train"]:
               print("\n=========== Now, Start {}===========".format(phase))
               
               model.train() if phase is "train" else model.eval()
               
               if args.baseline == True and args.baseline_type == 'clycle':
                    loss_d, loss_kl, loss_style, ndm_loss, loss_xt_reconstruction, loss_total = train_cycle(model,
                                                                                                                  epoch,
                                                                                                                  phase,
                                                                                                                  optimizer_g,
                                                                                                                  optimizer_d_A,
                                                                                                                  optimizer_d_B,
                                                                                                                  dataloaders[
                                                                                                                       phase],
                                                                                                                  args)
               else:
                    # except the clycle,  use this loop
                    loss_d,loss_kl,loss_style, ndm_loss, loss_total, loss_xt_reconstruction = train(model, epoch, phase, optimizer_g,  optimizer_d,dataloaders[phase], args)
               losses[phase][epoch] = torch.tensor([loss_d,loss_kl,loss_style, ndm_loss, loss_total, loss_xt_reconstruction,0,0])
          StepLR.step()

          # validation loop
          if  epoch != 0 and epoch % args.validate_epoch ==0 and args.validate == True: # epoch != 0 and epoch % args.validate_epoch ==0 and
               model.eval()
               data_eval = dataloaders['val']
               full_image_set = []
               low_image_set = []
               predict_image_set = []
               xt_imge_set = []
               batch = data_eval
               time_batch_start = time.time()
               full_image = batch["full_image"]

               full_image_set.append(full_image)
               quarter_image = batch["quarter_image"]
               total_test = quarter_image.shape[0]
               low_image_set.append(quarter_image)
               for index in tqdm(range(total_test)):
                    with torch.no_grad():
                         quarter_image_test = torch.tensor(quarter_image[index]).float()

                         quarter_image_test = torch.reshape(quarter_image_test, [1, 1, 512, 512])
                         if args.use_cuda:

                              quarter_image_test = Variable(quarter_image_test).cuda()
                         else:

                              pass
                         quarter_image_test = Variable(quarter_image_test)

                         predicts,xt_self = model.predict(quarter_image_test)
                         predict_image_set.append(predicts.cpu().detach().numpy())
                         xt_imge_set.append(xt_self.cpu().detach().numpy())
               full_image_set = np.concatenate(full_image_set,axis=0)
               full_image_set = full_image_set.reshape([-1,512,512,1])
               predict_image_set = np.concatenate(predict_image_set,axis=0).reshape([-1,512,512,1])
               xt_imge_set = np.concatenate(xt_imge_set, axis=0).reshape([-1, 512, 512, 1])
               low_image_set = np.concatenate(low_image_set,axis=0).reshape([-1,512,512,1])

               visualize(args,epoch,low_image_set,full_image_set,predict_image_set,xt_imge_set)
               torch.cuda.empty_cache()

               model.train()
               if args.baseline_type == 'clycle':
                     torch.save(model.netG_A.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))
               else:
                    torch.save(model.generator.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))

          # output bayesian coral
          if epoch != 0 and args.validate_coral == True:
               xs = args.test_patch_source
               xt = args.test_patch_target
               ndm_total_indomain = 0
               ndm_total_outdomain = 0
               model.eval()
               with torch.no_grad():

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
          data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in losses.items()}
          sio.savemat(args.loss_path + "/{}_losses.mat".format(args.name), mdict = data_save)
          print('-' * 20, 'D Loss:{:.6f}'.format(losses[phase][epoch][0].item()),'MSE Loss:{:.6f}'.format(losses[phase][epoch][5].item()),'-'*20)
          print("Time for epoch {} : {:.4f}min".format(epoch+1, (time.time()-time_epoch_start)/60))
          print("Time for ALL : {:.4f}h\n".format((time.time()-time_all_start)/3600))

     ##********************************************************************************************************************

     print("\nTrain Completed!! Time for ALL : {:.4f}h".format((time.time()-time_all_start)/3600))
     number = len(psnr)
     return sum(psnr)/number, sum(ssim)/number, sum(vif) / number, sum(vis)/number






def visualize(args,epoch,low,full,predict,xt_self):

     mean_psnr, mean_ssim,mean_vif,mean_vis,mean_psnr_ldct, mean_ssim_ldct,mean_vif_ldct,mean_vis_ldct = measure(args,low,full, predict)
     if args.validate:
          fig_titles = ['LDCT', 'Denoised by Ours', 'NDCT']
          plt.figure()
          f, axs = plt.subplots(figsize=(40, 40), nrows=1, ncols=4)
          if args.target == 'ISICDM':
               for num in range(0, 3):
                    ldct = np.squeeze(low[num])
                    xt = np.squeeze(xt_self[num])
                    ndct = np.squeeze(full[num])
                    denoising = np.squeeze(predict[num])
                    denoising = np.clip(denoising, 0, 1.)

                    for s in range(1):
                         ldct = windowing(renormalization(ldct, -1000, 400), -160, 240)
                         axs[0].imshow(ldct, cmap=plt.cm.gray)
                         axs[0].axis('off')

                         denoising = windowing(renormalization(denoising, -1000, 400), -160, 240)
                         axs[1].imshow(denoising, cmap=plt.cm.gray)
                         axs[1].axis('off')

                         # imshow
                         ndct = windowing(renormalization(ndct, -1000, 400), -160, 240)
                         axs[2].imshow(ndct, cmap=plt.cm.gray)
                         axs[2].axis('off')

                         xt = windowing(renormalization(xt, -1000, 400), -160, 240)
                         axs[3].imshow(xt, cmap=plt.cm.gray)
                         axs[3].axis('off')

                         plt.savefig(args.show_image + '/test_result_{}_{}.png'.format(num, epoch),
                                     bbox_inches='tight', pad_inches=0.0)

          else:
               for num in range(432,434):
                    ldct = np.squeeze(low[num])
                    ndct = np.squeeze(full[num])
                    xt = np.squeeze(xt_self[num])
                    denoising = np.squeeze(predict[num])
                    denoising = np.clip(denoising,0,1.)

                    for s in range(1):
                         ldct = windowing(renormalization(ldct,-1000,400),-160,240)
                         axs[0].imshow(ldct, cmap=plt.cm.gray)

                         axs[0].axis('off')

                         denoising = windowing(renormalization(denoising, -1000, 400), -160, 240)
                         axs[1].imshow(denoising, cmap=plt.cm.gray)
                         axs[1].axis('off')

                         # imshow
                         ndct = windowing(renormalization(ndct, -1000, 400), -160, 240)
                         axs[2].imshow(ndct, cmap=plt.cm.gray)
                         axs[2].axis('off')

                         xt = windowing(renormalization(xt, -1000, 400), -160, 240)
                         axs[3].imshow(xt, cmap=plt.cm.gray)
                         axs[3].axis('off')
                         plt.savefig(args.show_image + '/test_result_{}_{}.png'.format(num,epoch),
                                     bbox_inches='tight', pad_inches=0.0)
                         #plt.show()

     log_file = open(args.measure_path + '/{}_measure.txt'.format(args.name), 'a')
     log_file.write('p_d:{:.4f},ssim_d:{:.5f},vif_d:{:.5f},vis_d:{:.5f}, p_ldct:{:.4f},ssim_ldct:{:.5f},vif_ldct:{:.5f},vis_ldct:{:.5f} \n'.format(mean_psnr, mean_ssim,mean_vif,mean_vis,mean_psnr_ldct, mean_ssim_ldct,mean_vif_ldct,mean_vis_ldct))

     log_file.close()
     return mean_psnr, mean_ssim, mean_vif, mean_vis


