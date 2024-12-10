'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-17 15:39:21
@LastEditors: GuoYi
'''
import os
import torch
import time 
import numpy as np 
import torch.distributed as dist
import glob
import os
import shutil
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


def save_file(target_dir):
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)


## Check the path
##***********************************************************************************************************
def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)

def chaeck_dir_all(dirs):
    for dir in dirs:
        check_dir(dir)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)  # 
    torch.cuda.manual_seed(
        args.seed)  # 
    torch.cuda.manual_seed_all(
        args.seed)  # 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
## Many GPU training
##***********************************************************************************************************
def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"   # 断言函数 raise if not
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus

    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
        print("ngpus:",ngpus)
    elif ngpus == 1:
        model = model.cuda()
    return model


## Updata old model
##***********************************************************************************************************
def updata_model(model, args):
     print("Please set the path of expected model!")
     time.sleep(3)
     old_modle_path = args.old_modle_path
     model_reload_path = old_modle_path + "/" + args.old_modle_name + ".pkl"

     if os.path.isfile(model_reload_path):
          print("Loading previously trained network...")
          print("Load model:{}".format(model_reload_path))
          checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
          model_dict = model.state_dict()
          checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
          model_dict.update(checkpoint)
          model.load_state_dict(model_dict)
          del checkpoint
          torch.cuda.empty_cache()
          if args.use_cuda:
               model = model.cuda()
          print("Done Reload!")
     else:
          print("Can not reload model..../n")
          time.sleep(10)
          sys.exit(0)
     return model


## Updata ae
##***********************************************************************************************************
def updata_ae(model, ae_path):
     print("\n......Please set the path of AE!......")
     if os.path.isfile(ae_path):
          print("Ae set done, Loading...")
          print("Load model:{}".format(ae_path))
          checkpoint = torch.load(ae_path, map_location = lambda storage, loc: storage)
          model_dict = model.state_dict()
          checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
          model_dict.update(checkpoint)
          model.load_state_dict(model_dict)
          del checkpoint
          torch.cuda.empty_cache()
          if torch.cuda.is_available():
               model = model.cuda()
          print("Ae Reload!\n")
     else:
          print("Can not reload Ae....\n")
          time.sleep(10)
          sys.exit(0)
     return model


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
        checkpoint = torch.load(model_reload_path, map_location=lambda storage, loc: storage)
        model_dict = model.generator.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
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

    return model, optimizer_g, optimizer_d

import numpy as np


def im2col(img, k, stride=1):
    # Parameters
    m, n = img.shape
    s0, s1 = img.strides
    nrows = m - k + 1
    ncols = n - k + 1
    shape = (k, k, nrows, ncols)
    arr_stride = (s0, s1, s0, s1)

    ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
    return ret[:, :, ::stride, ::stride].reshape(k*k, -1)


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def moments(x, y, k, stride):
    kh = kw = k

    k_norm = k**2

    x_pad = np.pad(x, int((kh - stride)/2), mode='reflect')
    y_pad = np.pad(y, int((kw - stride)/2), mode='reflect')

    int_1_x = integral_image(x_pad)
    int_1_y = integral_image(y_pad)

    int_2_x = integral_image(x_pad*x_pad)
    int_2_y = integral_image(y_pad*y_pad)

    int_xy = integral_image(x_pad*y_pad)

    mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride])/k_norm
    mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride])/k_norm

    var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride])/k_norm - mu_x**2
    var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride])/k_norm - mu_y**2

    cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride])/k_norm - mu_x*mu_y

    mask_x = (var_x < 0)
    mask_y = (var_y < 0)

    var_x[mask_x] = 0
    var_y[mask_y] = 0

    cov_xy[mask_x + mask_y] = 0

    return (mu_x, mu_y, var_x, var_y, cov_xy)


def vif_gsm_model(pyr, subband_keys, M):
    tol = 1e-15
    s_all = []
    lamda_all = []

    for subband_key in subband_keys:
        y = pyr[subband_key]
        y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
        y = y[:y_size[0], :y_size[1]]

        y_vecs = im2col(y, M, 1)
        cov = np.cov(y_vecs)
        lamda, V = np.linalg.eigh(cov)
        lamda[lamda < tol] = tol
        cov = V@np.diag(lamda)@V.T

        y_vecs = im2col(y, M, M)

        s = np.linalg.inv(cov)@y_vecs
        s = np.sum(s * y_vecs, 0)/(M*M)
        s = s.reshape((int(y_size[0]/M), int(y_size[1]/M)))

        s_all.append(s)
        lamda_all.append(lamda)

    return s_all, lamda_all


def vif_channel_est(pyr_ref, pyr_dist, subband_keys, M):
    tol = 1e-15
    g_all = []
    sigma_vsq_all = []

    for i, subband_key in enumerate(subband_keys):
        y_ref = pyr_ref[subband_key]
        y_dist = pyr_dist[subband_key]

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1

        y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
        y_ref = y_ref[:y_size[0], :y_size[1]]
        y_dist = y_dist[:y_size[0], :y_size[1]]

        mu_x, mu_y, var_x, var_y, cov_xy = moments(y_ref, y_dist, winsize, M)

        g = cov_xy / (var_x + tol)
        sigma_vsq = var_y - g*cov_xy

        g[var_x < tol] = 0
        sigma_vsq[var_x < tol] = var_y[var_x < tol]
        var_x[var_x < tol] = 0

        g[var_y < tol] = 0
        sigma_vsq[var_y < tol] = 0

        sigma_vsq[g < 0] = var_y[g < 0]
        g[g < 0] = 0

        sigma_vsq[sigma_vsq < tol] = tol

        g_all.append(g)
        sigma_vsq_all.append(sigma_vsq)

    return g_all, sigma_vsq_all


def vif(img_ref, img_dist, wavelet='steerable', full=False):
    assert wavelet in ['steerable', 'haar', 'db2', 'bio2.2'], 'Invalid choice of wavelet'
    M = 3
    sigma_nsq = 0.1

    if wavelet == 'steerable':
        from pyrtools.pyramids import SteerablePyramidSpace as SPyr
        pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
        pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
        subband_keys = []
        for key in list(pyr_ref.keys())[1:-2:3]:
            subband_keys.append(key)
    else:
        from pywt import wavedec2
        ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
        ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
        pyr_ref = {}
        pyr_dist = {}
        subband_keys = []
        for i in range(4):
            pyr_ref[(3-i, 0)] = ret_ref[i+1][0]
            pyr_ref[(3-i, 1)] = ret_ref[i+1][1]
            pyr_dist[(3-i, 0)] = ret_dist[i+1][0]
            pyr_dist[(3-i, 1)] = ret_dist[i+1][1]
            subband_keys.append((3-i, 0))
            subband_keys.append((3-i, 1))
        pyr_ref[4] = ret_ref[0]
        pyr_dist[4] = ret_dist[0]

    subband_keys.reverse()
    n_subbands = len(subband_keys)

    [g_all, sigma_vsq_all] = vif_channel_est(pyr_ref, pyr_dist, subband_keys, M)

    [s_all, lamda_all] = vif_gsm_model(pyr_ref, subband_keys, M)

    nums = np.zeros((n_subbands,))
    dens = np.zeros((n_subbands,))
    for i in range(n_subbands):
        g = g_all[i]
        sigma_vsq = sigma_vsq_all[i]
        s = s_all[i]
        lamda = lamda_all[i]

        n_eigs = len(lamda)

        lev = int(np.ceil((i+1)/2))
        winsize = 2**lev + 1
        offset = (winsize - 1)/2
        offset = int(np.ceil(offset/M))

        g = g[offset:-offset, offset:-offset]
        sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
        s = s[offset:-offset, offset:-offset]

        for j in range(n_eigs):
            nums[i] += np.mean(np.log(1 + g*g*s*lamda[j]/(sigma_vsq+sigma_nsq)))
            dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

    if not full:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
    else:
        return np.mean(nums + 1e-4)/np.mean(dens + 1e-4), (nums + 1e-4), (dens + 1e-4)


def vif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    mu_x, mu_y, var_x, var_y, cov_xy = moments(x, y, k, stride)

    g = cov_xy / (var_x + 1e-10)
    sv_sq = var_y - g * cov_xy

    g[var_x < 1e-10] = 0
    sv_sq[var_x < 1e-10] = var_y[var_x < 1e-10]
    var_x[var_x < 1e-10] = 0

    g[var_y < 1e-10] = 0
    sv_sq[var_y < 1e-10] = 0

    sv_sq[g < 0] = var_x[g < 0]
    g[g < 0] = 0
    sv_sq[sv_sq < 1e-10] = 1e-10

    vif_val = np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4)
    if (full):
        # vif_map = (np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4)/(np.log(1 + var_x / sigma_nsq) + 1e-4)
        # return (vif_val, vif_map)
        return (np.sum(np.log(1 + g**2 * var_x / (sv_sq + sigma_nsq)) + 1e-4), np.sum(np.log(1 + var_x / sigma_nsq) + 1e-4), vif_val)
    else:
        return vif_val


def msvif_spatial(img_ref, img_dist, k=11, sigma_nsq=0.1, stride=1, full=False):
    x = img_ref.astype('float32')
    y = img_dist.astype('float32')

    n_levels = 5
    nums = np.ones((n_levels,))
    dens = np.ones((n_levels,))
    for i in range(n_levels-1):
        if np.min(x.shape) <= k:
            break
        nums[i], dens[i], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
        x = x[:(x.shape[0]//2)*2, :(x.shape[1]//2)*2]
        y = y[:(y.shape[0]//2)*2, :(y.shape[1]//2)*2]
        x = (x[::2, ::2] + x[1::2, ::2] + x[1::2, 1::2] + x[::2, 1::2])/4
        y = (y[::2, ::2] + y[1::2, ::2] + y[1::2, 1::2] + y[::2, 1::2])/4

    if np.min(x.shape) > k:
        nums[-1], dens[-1], _ = vif_spatial(x, y, k, sigma_nsq, stride, full=True)
    msvifval = np.sum(nums) / np.sum(dens)

    if full:
        return msvifval, nums, dens
    else:
        return msvifval
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


def measure(args, low, full, predict):
    psnr = []
    ssim = []
    GMSD = []
    DSS = []

    psnr_ldct = []
    ssim_ldct = []
    GMSD_ldct = []
    DSS_ldct = []

    total_imges = low.shape[0]

    if args.target == 'ISICDM':

        top1 = [295, 220, 200]
        top2 = [415, 400, 400]
        bottom1 = [180, 100, 300]
        bottom2 = [350, 450, 450]
        predict = np.clip(predict, 0, 1)
        for i in range(low.shape[0]):
            denoising_test = renormalization(predict[i, top1[i]:top2[i], bottom1[i]:bottom2[i]], -1000, 400)
            full_test = renormalization(full[i, top1[i]:top2[i], bottom1[i]:bottom2[i]], -1000, 400)
            low_test = renormalization(low[i, top1[i]:top2[i], bottom1[i]:bottom2[i]], -1000, 400)

            psnr.append(np.mean(denoising_test))
            ssim.append(np.mean(full_test))
            GMSD.append(np.std(denoising_test))
            DSS.append(np.std(full_test))

            psnr_ldct.append(np.mean(denoising_test))
            ssim_ldct.append(np.mean(full_test))
            GMSD_ldct.append(np.std(denoising_test))
            DSS_ldct.append(np.std(full_test))

        return mean(psnr), mean(ssim), mean(GMSD), mean(DSS),mean(psnr_ldct), mean(ssim_ldct), mean(GMSD_ldct), mean(DSS_ldct),

    if args.validate:
        predict = np.clip(predict, 0, 1)
        denoising_test = torch.tensor(predict[:, 50:450, 50:450]).cuda()
        full_test = torch.tensor(full[:, 50:450, 50:450]).cuda()
        low_test = torch.tensor(low[:, 50:450, 50:450]).cuda()

        for i in tqdm(range(0, total_imges)):
            psnr.append(
                piq.psnr(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
            ssim.append(
                piq.ssim(denoising_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
            GMSD.append(piq.GMSDLoss(data_range=1.)(denoising_test[i].resize(1, 1, 400, 400),
                                                    full_test[i].resize(1, 1, 400, 400)).data.cpu())
            DSS.append(piq.DSSLoss(data_range=1.)(denoising_test[i].resize(1, 1, 400, 400),
                                                  full_test[i].resize(1, 1, 400, 400)).data.cpu())

            psnr_ldct.append(
                piq.psnr(low_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
            ssim_ldct.append(
                piq.ssim(low_test[i].resize(1, 1, 400, 400), full_test[i].resize(1, 1, 400, 400)).data.cpu())
            GMSD_ldct.append(piq.GMSDLoss(data_range=1.)(low_test[i].resize(1, 1, 400, 400),
                                                    full_test[i].resize(1, 1, 400, 400)).data.cpu())
            DSS_ldct.append(piq.DSSLoss(data_range=1.)(low_test[i].resize(1, 1, 400, 400),
                                                  full_test[i].resize(1, 1, 400, 400)).data.cpu())


            '''
            psnr.append(piq.psnr(denoising_test[i].resize(1, 1, 512, 512), full_test[i].resize(1, 1, 512, 512)).data.cpu())
            ssim.append(piq.ssim(denoising_test[i].resize(1, 1, 512, 512), full_test[i].resize(1, 1, 512, 512)).data.cpu())
            GMSD.append(piq.GMSDLoss(data_range=1.)(denoising_test[i].resize(1, 1, 512, 512), full_test[i].resize(1, 1, 512, 512)).data.cpu())
            DSS.append(piq.DSSLoss(data_range=1.)(denoising_test[i].resize(1, 1, 512, 512), full_test[i].resize(1, 1, 512, 512)).data.cpu())
            '''
        return mean(psnr), mean(ssim), mean(GMSD), mean(DSS),mean(psnr_ldct), mean(ssim_ldct), mean(GMSD_ldct), mean(DSS_ldct),

    else:
        predict = np.clip(predict, 0, 1)
        if args.validate_region == 'head':
            denoising_test = torch.tensor(predict[:, :, 100:400, 100:400]).cuda()
            full_test = torch.tensor(full[:, :, 100:400, 100:400]).cuda()
        else:
            denoising_test = torch.tensor(predict[:, :, 50:400, 50:450]).cuda()
            full_test = torch.tensor(full[:, :, 50:400, 50:450]).cuda()
        psnr_mean = piq.psnr(denoising_test, full_test, reduction='mean').data.cpu()
        ssim_mean = piq.ssim(denoising_test, full_test, reduction='mean').data.cpu()
        for i in tqdm(range(total_imges)):
            if args.validate_region == 'head':
                vif_.append(piq.vif_p(denoising_test[i].resize(1, 1, 300, 300), full_test[i].resize(1, 1, 300, 300),
                                      reduction='mean').data.cpu())
                vis.append(piq.VSILoss(data_range=1.)(denoising_test[i].resize(1, 1, 300, 300),
                                                      full_test[i].resize(1, 1, 300, 300)).data.cpu())
            else:
                vif_.append(piq.vif_p(denoising_test[i].resize(1, 1, 350, 400), full_test[i].resize(1, 1, 350, 400),
                                      reduction='mean').data.cpu())
                vis.append(piq.VSILoss(data_range=1.)(denoising_test[i].resize(1, 1, 350, 400),
                                                      full_test[i].resize(1, 1, 350, 400)).data.cpu())
        return psnr_mean, ssim_mean, mean(GMSD), mean(vis)

    return mean(psnr), mean(ssim), mean(vif_), mean(DSS)

def renormalization(data,min_hu,max_hu):

    output = data*(max_hu - min_hu) + min_hu
    return output

def windowing(image,min_bound=-1000,max_bound=1000):
    output = (image-min_bound)/(max_bound-min_bound)
    output[output<0]=0
    output[output>1]=1
    return output